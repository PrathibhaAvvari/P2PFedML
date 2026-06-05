import socket
import threading
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import struct
from collections import defaultdict
import logging
import sys
import argparse

np.random.seed(42)
torch.manual_seed(42)


# --- Timing instrumentation (per-client) ---
_timing_lock = threading.Lock()
client_timing = defaultdict(lambda: {
    "training_s": 0.0,
    "send_s": 0.0,
    "recv_s": 0.0,
    "comm_phase_s": 0.0,
    "wait_s": 0.0,
})
_net_lock = threading.Lock()
client_net_io = defaultdict(lambda: {
    "bytes_sent": 0,
    "bytes_recv": 0,
    "messages_sent": 0,
    "messages_recv": 0,
})


def _add_timing(client_id, key, delta_s):
    if client_id is None:
        return
    if delta_s is None or delta_s <= 0:
        return
    with _timing_lock:
        client_timing[client_id][key] += float(delta_s)


def _add_net_io(client_id, bytes_sent=0, bytes_recv=0, messages_sent=0, messages_recv=0):
    if client_id is None:
        return
    with _net_lock:
        stats = client_net_io[client_id]
        if bytes_sent > 0:
            stats["bytes_sent"] += int(bytes_sent)
        if bytes_recv > 0:
            stats["bytes_recv"] += int(bytes_recv)
        if messages_sent > 0:
            stats["messages_sent"] += int(messages_sent)
        if messages_recv > 0:
            stats["messages_recv"] += int(messages_recv)


def _bytes_to_gb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 ** 3)

BATCH_SIZE = 32
EPOCHS_PER_ROUND = 5
THRESHOLD = 0.6  # Threshold for weight difference between rounds
DEVICE = torch.device("cpu")
TIMEOUT = 25  # Timeout in seconds for waiting for models
CONNECT_TIMEOUT = 3.0  # Timeout in seconds for peer connect/send operations
TCP_RETRIES = 3  # Number of retries for peer send attempts
R_PRIME = 100  # Maximum number of rounds
MINIMUM_ROUNDS = 40  # Minimum rounds before checking termination criteria
COUNT_THRESHOLD = 5  # Number of consecutive rounds for weight difference and no crashes
LR_DECAY_EVERY = 15
LR_DECAY_GAMMA = 0.5

# Async-A settings (round counter remains, communication is best-effort and non-blocking beyond a small budget)
ASYNC_COMM_BUDGET_S = 3.0

# Gradient + frequency layer-sharing hyperparameters (push-based broadcast)
LAYER_SELECTION_L = 18
LAYER_SELECTION_X_PERCENT = 0.0
TOTAL_LOGICAL_LAYERS = None

# Adaptive crash detection with recovery
CRASH_THRESHOLD = 5  # Number of consecutive misses before marking as crashed

# Model selection: 'SimpleCNN', 'SimpleCNN10', 'VGG11BN', 'VGG13BN', 'VGG16BN'
MODEL_NAME = 'SimpleCNN10'  # Change this to use different models


def send_message(conn, message, owner_id=None):
    data = pickle.dumps(message)
    message_length = struct.pack('!I', len(data))
    payload = message_length + data
    conn.sendall(payload)
    _add_net_io(owner_id, bytes_sent=len(payload), messages_sent=1)


def receive_message(conn, owner_id=None):
    message_length_data = conn.recv(4)
    if not message_length_data:
        return None
    message_length = struct.unpack('!I', message_length_data)[0]
    data = b''
    while len(data) < message_length:
        part = conn.recv(min(4096, message_length - len(data)))
        if not part:
            return None
        data += part
    _add_net_io(owner_id, bytes_recv=4 + len(data), messages_recv=1)
    return pickle.loads(data)


def parse_input_file():
    try:
        with open("inputf.txt", "r") as file:
            lines = file.read().splitlines()
            if len(lines) < 4:
                raise ValueError("Input file does not contain enough lines.")

            num_clients, num_machines = map(int, lines[0].split())
            current_machine_ip = lines[1].strip()
            all_ips = [ip.strip() for ip in lines[2].split(",")]
            num_faults = int(lines[3])
            faults = []

            if len(lines) < 4 + num_faults:
                raise ValueError(
                    f"Input file does not contain enough lines for the specified number of faults ({num_faults})."
                )

            for i in range(num_faults):
                id, fr, y = map(int, lines[4 + i].split(','))
                faults.append((id, fr, y))
        return num_clients, num_machines, current_machine_ip, all_ips, faults
    except FileNotFoundError:
        print("The input file was not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None, None, None


NUM_CLIENTS, NUM_MACHINES, CURRENT_MACHINE_IP, ips, faults = parse_input_file()

if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    exit(1)

# Configure the logger with a dynamic filename based on input parameters
logger = logging.getLogger('federated_learning')
logger.setLevel(logging.INFO)

# Create a formatter for log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Dynamically name the log file
log_filename = f"min40_crash_test_{TIMEOUT}_gradfreq_push_asyncA_log_{NUM_CLIENTS}_{NUM_MACHINES}_{len(faults)}.txt"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


# Custom filter to log only crash-related messages to the file
class CrashFilter(logging.Filter):
    def filter(self, record):
        return "crash" in record.msg.lower() or "crashing" in record.msg.lower()


# Add filter to file handler only (not console)
file_handler.addFilter(CrashFilter())

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Redirect print statements to the logger
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


# Redirect stdout to the logger
sys.stdout = LoggerWriter(logger, logging.INFO)

retries_list = [TCP_RETRIES] * NUM_CLIENTS
adj = [[j for j in range(NUM_CLIENTS) if j != i] for i in range(NUM_CLIENTS)]
terminate_messages = [0] * NUM_CLIENTS
model_messages = [0] * NUM_CLIENTS

# CIFAR-10 dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indices = np.random.permutation(len(train_dataset))


def create_dirichlet_non_iid_splits(dataset, num_clients, alpha=0.5):
    num_classes = 10
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}

    for c, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(indices)).astype(int)
        start_idx = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(indices[start_idx:start_idx + count])
            start_idx += count

    for client_id, indices in client_indices.items():
        np.random.shuffle(indices)

    client_data = [torch.utils.data.Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return client_data


client_data = create_dirichlet_non_iid_splits(
    train_dataset, NUM_CLIENTS, alpha=0.5
)

msg_lck = threading.Lock()
latest_models_lock = threading.Lock()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            # Block 2: 128 channels
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            # Block 3: 256 channels
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            # Block 4: 512 channels
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _make_vgg_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(True),
            ])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def VGG11BN():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG13BN():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG16BN():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def get_model(model_name):
    """Factory function to create model based on name."""
    if model_name == 'SimpleCNN':
        return SimpleCNN()
    elif model_name == 'SimpleCNN10':
        return SimpleCNN10()
    elif model_name == 'VGG11BN':
        return VGG11BN()
    elif model_name == 'VGG13BN':
        return VGG13BN()
    elif model_name == 'VGG16BN':
        return VGG16BN()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: SimpleCNN, SimpleCNN10, VGG11BN, VGG13BN, VGG16BN")


def _state_dict_to_numpy(model: nn.Module):
    sd = model.state_dict()
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def _numpy_to_state_dict_torch(state_np):
    # Preserve exact keys; tensors are created on DEVICE.
    return {k: torch.tensor(v).to(DEVICE) for k, v in state_np.items()}


def _logical_layer_key(param_name: str) -> str:
    # Groups e.g. conv1.weight + conv1.bias under conv1
    if '.' not in param_name:
        return param_name
    return param_name.rsplit('.', 1)[0]


def _group_params_by_logical_layer(state_np):
    groups = defaultdict(list)
    for name in state_np.keys():
        groups[_logical_layer_key(name)].append(name)
    return dict(groups)


def _compute_total_logical_layers(model_name: str) -> int:
    model = get_model(model_name).to(DEVICE)
    state_np = _state_dict_to_numpy(model)
    return len(_group_params_by_logical_layer(state_np))


def _compute_logical_layer_gradient_scores(model: nn.Module, layer_groups):
    grad_scores = {layer_key: 0.0 for layer_key in layer_groups.keys()}
    for pname, param in model.named_parameters():
        if param.grad is None:
            continue
        layer_key = _logical_layer_key(pname)
        if layer_key in grad_scores:
            grad_scores[layer_key] += float(param.grad.detach().norm().item())
    return grad_scores


def _weighted_mean(values, weights):
    if not values:
        return None
    total = float(np.sum(weights))
    if total <= 0:
        total = float(len(weights))
        weights = [1.0] * len(weights)
    acc = np.zeros_like(values[0], dtype=np.float64)
    for v, w in zip(values, weights):
        acc += v * float(w)
    return acc / total


def _select_layers_by_grad_and_frequency(layer_groups, grad_scores, layer_share_frequency, l_count, x_percent):
    all_layers = list(layer_groups.keys())
    if not all_layers:
        return []

    l_eff = min(max(0, int(l_count)), len(all_layers))
    if l_eff == 0:
        return []

    x_eff = float(np.clip(x_percent, 0.0, 100.0))
    grad_pick_count = int(round((x_eff / 100.0) * l_eff))
    grad_pick_count = min(max(0, grad_pick_count), l_eff)

    grad_sorted = sorted(
        all_layers,
        key=lambda layer_key: (
            -float(grad_scores.get(layer_key, 0.0)),
            int(layer_share_frequency.get(layer_key, 0)),
            layer_key,
        ),
    )
    grad_selected = grad_sorted[:grad_pick_count]

    remaining_needed = l_eff - len(grad_selected)
    remaining_candidates = [k for k in all_layers if k not in set(grad_selected)]
    freq_selected = sorted(
        remaining_candidates,
        key=lambda layer_key: (
            int(layer_share_frequency.get(layer_key, 0)),
            -float(grad_scores.get(layer_key, 0.0)),
            layer_key,
        ),
    )[:remaining_needed]

    return grad_selected + freq_selected


def broadcast_selected_layers(id, selected_layer_params, current_round, ips, n_samples):
    """Broadcast only selected layer parameters to all neighbors."""
    global model_messages
    message = {
        'type': 'selected_layers',
        'id': id,
        'round': current_round,
        'layers': selected_layer_params,
        'n_samples': int(n_samples),
    }
    for pid in adj[id]:
        with msg_lck:
            model_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)


def _state_dict_to_list_sorted(state_np):
    # Stable ordering for similarity checks.
    return [state_np[k] for k in sorted(state_np.keys())]


def models_are_similar_list(weights1_list, weights2_list, threshold):
    for w1, w2 in zip(weights1_list, weights2_list):
        norm = np.linalg.norm(w1 - w2)
        if norm > threshold:
            return False
    return True


def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def tcp_client(id, target_id, target_ip, message):
    global retries_list
    retries = TCP_RETRIES
    while retries > 0:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.settimeout(CONNECT_TIMEOUT)
            client.connect((target_ip, 8650 + target_id))
            send_message(client, message, owner_id=id)
            break
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, socket.timeout, OSError):
            retries -= 1
            retries_list[target_id] -= 1
            if retries == 0:
                break
            time.sleep(1)
        finally:
            client.close()


def broadcast_weights(id, weights_state_np, current_round, terminate, ips, latest_models, crash_away_list, prev_list):
    global model_messages
    message = {'type': 'weights', 'weights': weights_state_np, 'round': current_round, 'terminate': terminate, 'id': id}
    for pid in adj[id]:
        with msg_lck:
            model_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)
    latest_models[id] = weights_state_np


def broadcast_terminate(id, ips):
    global terminate_messages
    message = {'type': 'terminate'}
    for pid in adj[id]:
        terminate_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)


def tcp_server(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list, stop_event, received_layers_buffer):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((local_ip, 8650 + id))
    server.listen(NUM_CLIENTS - 1)
    server.settimeout(1.0)

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        _t_recv0 = time.perf_counter()
        msg = receive_message(conn, owner_id=id)
        _add_timing(id, "recv_s", time.perf_counter() - _t_recv0)
        if msg is None:
            conn.close()
            continue
        if msg['type'] == 'terminate':
            terminate_flags.append(1)
            conn.close()
            break
        if msg['type'] == 'selected_layers':
            sender_id = msg.get('id')
            round_id = msg.get('round')
            layer_params = msg.get('layers', {})
            with latest_models_lock:
                received_layers_buffer.append({
                    'sender_id': sender_id,
                    'round': round_id,
                    'layers': layer_params,
                    'n_samples': msg.get('n_samples', None),
                })
        conn.close()
    server.close()


def client_logic(id, local_ip, ips, faults, timing_store):
    model = get_model(MODEL_NAME).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    local_n_samples = len(client_data[id])

    # Core tracking variables
    current_round = 0
    crash_away_list = [False] * NUM_CLIENTS
    layer_share_frequency = defaultdict(int)
    consecutive_misses = defaultdict(int)
    received_layers_buffer = []
    
    # Termination tracking
    previous_weights_list = None
    counter = 0
    crash_counter = 0
    crashed_in_rounds = []
    terminate_flags = []
    best_accuracy = -1.0
    best_round = -1
    final_accuracy = 0.0
    
    # TCP server for receiving layers
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=tcp_server,
        args=(id, [], terminate_flags, local_ip, {}, crash_away_list, [], stop_event, received_layers_buffer),
    )
    server_thread.start()
    time.sleep(2)

    while current_round < R_PRIME:
        if current_round > 0 and (current_round % LR_DECAY_EVERY == 0):
            for param_group in optimizer.param_groups:
                param_group["lr"] *= LR_DECAY_GAMMA
        _t_train0 = time.perf_counter()
        model.train()
        for epoch in range(EPOCHS_PER_ROUND):
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        _add_timing(id, "training_s", time.perf_counter() - _t_train0)

        local_state_np = _state_dict_to_numpy(model)
        layer_groups = _group_params_by_logical_layer(local_state_np)

        # Check if this client should crash
        for fault in faults:
            if fault[0] == id and fault[1] == current_round:
                print(f"Client {id} is crashing at round {current_round}")
                stop_event.set()
                server_thread.join(timeout=2)
                return

        # Check if termination flag is received from other clients
        if terminate_flags:
            print(f"Client {id} received termination flag at round {current_round}")
            broadcast_terminate(id, ips)
            break

        # --- Gradient+frequency layer selection + push-based broadcast ---
        grad_scores = _compute_logical_layer_gradient_scores(model, layer_groups)
        selected_layers = _select_layers_by_grad_and_frequency(
            layer_groups=layer_groups,
            grad_scores=grad_scores,
            layer_share_frequency=layer_share_frequency,
            l_count=LAYER_SELECTION_L,
            x_percent=LAYER_SELECTION_X_PERCENT,
        )

        # Build selected layer parameters to broadcast
        selected_layer_params = {}
        for layer_key in selected_layers:
            for param_name in layer_groups[layer_key]:
                selected_layer_params[param_name] = local_state_np[param_name]
            layer_share_frequency[layer_key] += 1

        # Broadcast selected layers to all neighbors
        _t_comm0 = time.perf_counter()
        broadcast_selected_layers(id, selected_layer_params, current_round, ips, local_n_samples)
        _add_timing(id, "send_s", time.perf_counter() - _t_comm0)

        # Semi-sync wait window: collect messages up to current round for up to TIMEOUT seconds.
        _t_wait0 = time.perf_counter()
        expected_peers = set(adj[id]) - {p for p in range(NUM_CLIENTS) if crash_away_list[p]}
        received_layers_this_round = []
        recv_deadline = time.time() + TIMEOUT
        poll_sleep_s = 0.05
        while time.time() < recv_deadline:
            with latest_models_lock:
                round_msgs = [msg for msg in received_layers_buffer if msg['round'] <= current_round]
                if round_msgs:
                    received_layers_this_round.extend(round_msgs)
                    received_layers_buffer[:] = [msg for msg in received_layers_buffer if msg['round'] > current_round]

            # Early exit: if we already have at least one update from each expected peer.
            if expected_peers:
                got_senders = {msg['sender_id'] for msg in received_layers_this_round}
                if expected_peers.issubset(got_senders):
                    break
            time.sleep(poll_sleep_s)
        _add_timing(id, "wait_s", time.perf_counter() - _t_wait0)

        # Deduplicate by sender: keep the newest (highest-round) message from each sender
        latest_by_sender = {}
        for msg in received_layers_this_round:
            sender_id = msg['sender_id']
            prev = latest_by_sender.get(sender_id)
            if prev is None or msg['round'] >= prev['round']:
                latest_by_sender[sender_id] = msg

        # Crash detection with recovery: track consecutive misses
        received_peers = set(latest_by_sender.keys())
        new_crashes = False

        for peer_id in range(NUM_CLIENTS):
            if peer_id == id:
                continue
            if peer_id in received_peers:
                consecutive_misses[peer_id] = 0
                if crash_away_list[peer_id]:
                    crash_away_list[peer_id] = False
            else:
                consecutive_misses[peer_id] += 1
                if consecutive_misses[peer_id] >= CRASH_THRESHOLD and not crash_away_list[peer_id]:
                    crash_away_list[peer_id] = True
                    new_crashes = True
                    print(
                        f"Client {id} detected crash/unreachable peer {peer_id} at round {current_round} "
                        f"(after {consecutive_misses[peer_id]} consecutive misses)"
                    )

        # Average received layers with local layers
        # For each parameter in selected layers, collect all received versions (one per sender)
        param_accumulator = defaultdict(list)
        param_weights = defaultdict(list)
        for sender_id, msg in latest_by_sender.items():
            sender_n = msg.get('n_samples')
            if sender_n is None:
                sender_n = 1.0
            for param_name, param_value in msg['layers'].items():
                param_accumulator[param_name].append(param_value)
                param_weights[param_name].append(float(sender_n))

        # Compute new state: average all received + own
        new_state_np = dict(local_state_np)
        for param_name, received_values in param_accumulator.items():
            if param_name in new_state_np and len(received_values) > 0:
                all_values = received_values + [local_state_np[param_name]]
                all_weights = param_weights[param_name] + [float(local_n_samples)]
                new_state_np[param_name] = _weighted_mean(all_values, all_weights)

        if new_crashes:
            crashed_in_rounds.append(current_round)
            crash_counter = 0
        else:
            crash_counter += 1

        # Load back to model
        model.load_state_dict(_numpy_to_state_dict_torch(new_state_np), strict=True)

        accuracy = compute_accuracy(model, test_loader)
        final_accuracy = float(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_round = current_round
        print(f"Client {id} - Round {current_round}: Accuracy: {accuracy:.2f}%, Selected {len(selected_layers)} layers, Received from {len(latest_by_sender)} unique peers")

        new_weights_list = _state_dict_to_list_sorted(new_state_np)

        # Termination check (same thresholds/logic, using stable list order)
        if current_round >= MINIMUM_ROUNDS:
            if previous_weights_list is not None and models_are_similar_list(new_weights_list, previous_weights_list, THRESHOLD):
                counter += 1
            else:
                counter = 0

            no_recent_crashes = True
            for r in range(current_round - COUNT_THRESHOLD + 1, current_round + 1):
                if r in crashed_in_rounds:
                    no_recent_crashes = False
                    break

            if counter >= COUNT_THRESHOLD and no_recent_crashes:
                print(
                    f"Client {id} met termination criteria at round {current_round}: "
                    f"stable weights for {COUNT_THRESHOLD} rounds and no crashes"
                )
                broadcast_weights(
                    id,
                    local_state_np,
                    current_round,
                    terminate=1,
                    ips=ips,
                    latest_models={},
                    crash_away_list=crash_away_list,
                    prev_list=[],
                )
                break

        previous_weights_list = new_weights_list
        current_round += 1

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_terminate(id, ips)

    print(f"Client {id} finished.")
    with _timing_lock:
        t = dict(client_timing.get(id, {}))
    with _net_lock:
        net = dict(client_net_io.get(id, {}))
    train_s = float(t.get("training_s", 0.0))
    send_s = float(t.get("send_s", 0.0))
    recv_s = float(t.get("recv_s", 0.0))
    comm_phase_s = float(t.get("comm_phase_s", 0.0))
    wait_s = float(t.get("wait_s", 0.0))
    comm_io_s = send_s + recv_s
    total_s = train_s + comm_io_s + comm_phase_s + wait_s
    bytes_sent = int(net.get("bytes_sent", 0))
    bytes_recv = int(net.get("bytes_recv", 0))
    messages_sent = int(net.get("messages_sent", 0))
    messages_recv = int(net.get("messages_recv", 0))
    print(
        f"Client {id} timing (s): train={train_s:.2f}, comm_io={comm_io_s:.2f} [send {send_s:.2f}, recv {recv_s:.2f}], comm_phase={comm_phase_s:.2f}, wait_timeout={wait_s:.2f}, comm_total={comm_io_s + comm_phase_s:.2f}, total={total_s:.2f}"
    )
    print(
        f"Client {id} bytes (app-level): sent={bytes_sent} ({_bytes_to_gb(bytes_sent):.3f} GB), "
        f"recv={bytes_recv} ({_bytes_to_gb(bytes_recv):.3f} GB), "
        f"messages_sent={messages_sent}, messages_recv={messages_recv}"
    )
    timing_store[id] = {
        "training_s": train_s,
        "comm_io_s": comm_io_s,
        "send_s": send_s,
        "recv_s": recv_s,
        "comm_phase_s": comm_phase_s,
        "wait_s": wait_s,
        "total_s": total_s,
        "bytes_sent": bytes_sent,
        "bytes_recv": bytes_recv,
        "messages_sent": messages_sent,
        "messages_recv": messages_recv,
        "final_acc": final_accuracy,
        "best_acc": best_accuracy,
        "best_round": best_round,
        "last_round": int(current_round),
    }
    broadcast_terminate(id, ips)
    stop_event.set()
    server_thread.join()


def main():
    global model_messages, terminate_messages
    start_time = time.time()
    l_share_pct = (100.0 * LAYER_SELECTION_L / TOTAL_LOGICAL_LAYERS) if TOTAL_LOGICAL_LAYERS else 0.0
    print(
        f"Starting Federated Learning (Gradient+Frequency Layer PUSH, Async, "
        f"Model={MODEL_NAME}, L={LAYER_SELECTION_L}/{TOTAL_LOGICAL_LAYERS} ({l_share_pct:.1f}%), "
        f"x={LAYER_SELECTION_X_PERCENT}%)"
    )

    threads = []
    timing_store = {}
    for i in range(NUM_CLIENTS):
        if ips[i] == str(CURRENT_MACHINE_IP):
            client_thread = threading.Thread(
                target=client_logic,
                args=(i, CURRENT_MACHINE_IP, ips, faults, timing_store),
            )
            threads.append(client_thread)
            client_thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    total_time = end_time - start_time

    total_model_messages = sum(model_messages)
    total_termination_messages = sum(terminate_messages)

    print("\nFederated Learning Completed")
    print("Current Machine IP:", CURRENT_MACHINE_IP)
    print("Number of Clients:", NUM_CLIENTS)
    print(f"Total layer requests sent: {total_model_messages}")
    print("Total Termination Messages Passed:", total_termination_messages)
    print(f"Total Time Taken: {total_time:.2f} seconds")

    local_client_ids = sorted([i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)])
    if local_client_ids:
        print("\nPer-client timing summary (seconds)")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            train_s = float(t.get("training_s", 0.0))
            comm_io_s = float(t.get("comm_io_s", 0.0))
            send_s = float(t.get("send_s", 0.0))
            recv_s = float(t.get("recv_s", 0.0))
            comm_phase_s = float(t.get("comm_phase_s", 0.0))
            wait_s = float(t.get("wait_s", 0.0))
            total_s = float(t.get("total_s", 0.0))
            bytes_sent = int(t.get("bytes_sent", 0))
            bytes_recv = int(t.get("bytes_recv", 0))
            print(
                f"  Client {cid}: train={train_s:.2f}, "
                f"comm_io={comm_io_s:.2f} [send {send_s:.2f}, recv {recv_s:.2f}], "
                f"comm_phase={comm_phase_s:.2f}, wait_timeout={wait_s:.2f}, "
                f"comm_total={comm_io_s + comm_phase_s:.2f}, "
                f"total={total_s:.2f}, bytes_sent={bytes_sent} ({_bytes_to_gb(bytes_sent):.3f} GB), "
                f"bytes_recv={bytes_recv} ({_bytes_to_gb(bytes_recv):.3f} GB)"
            )

        print("\nPer-client accuracy summary")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            final_acc = float(t.get("final_acc", 0.0))
            best_acc = float(t.get("best_acc", 0.0))
            best_rnd = int(t.get("best_round", -1))
            last_rnd = int(t.get("last_round", -1))
            print(
                f"  Client {cid}: final_acc={final_acc:.2f}%, "
                f"best_acc={best_acc:.2f}% (round {best_rnd}), "
                f"last_round={last_rnd}"
            )

        avg_final_acc = (
            sum(float(timing_store.get(cid, {}).get("final_acc", 0.0)) for cid in local_client_ids)
            / max(1, len(local_client_ids))
        )
        print(f"\nAverage final accuracy (local clients): {avg_final_acc:.2f}%")

        agg_train = sum(float(timing_store.get(cid, {}).get("training_s", 0.0)) for cid in local_client_ids)
        agg_comm_io = sum(float(timing_store.get(cid, {}).get("comm_io_s", 0.0)) for cid in local_client_ids)
        agg_comm_phase = sum(float(timing_store.get(cid, {}).get("comm_phase_s", 0.0)) for cid in local_client_ids)
        agg_wait = sum(float(timing_store.get(cid, {}).get("wait_s", 0.0)) for cid in local_client_ids)
        agg_total = sum(float(timing_store.get(cid, {}).get("total_s", 0.0)) for cid in local_client_ids)
        agg_bytes_sent = sum(int(timing_store.get(cid, {}).get("bytes_sent", 0)) for cid in local_client_ids)
        agg_bytes_recv = sum(int(timing_store.get(cid, {}).get("bytes_recv", 0)) for cid in local_client_ids)
        n = max(1, len(local_client_ids))
        print("\nAggregate timing (local clients)")
        print(
            f"  Sum  : train={agg_train:.2f}, comm_io={agg_comm_io:.2f}, "
            f"comm_phase={agg_comm_phase:.2f}, wait_timeout={agg_wait:.2f}, total={agg_total:.2f}"
        )
        print(
            f"  Avg  : train={agg_train/n:.2f}, comm_io={agg_comm_io/n:.2f}, "
            f"comm_phase={agg_comm_phase/n:.2f}, wait_timeout={agg_wait/n:.2f}, total={agg_total/n:.2f}"
        )
        print(
            f"  Bytes: sent_sum={agg_bytes_sent}, recv_sum={agg_bytes_recv}, "
            f"sent_avg={agg_bytes_sent // n}, recv_avg={agg_bytes_recv // n}, "
            f"sent_sum_gb={_bytes_to_gb(agg_bytes_sent):.3f}, recv_sum_gb={_bytes_to_gb(agg_bytes_recv):.3f}, "
            f"sent_avg_gb={_bytes_to_gb(agg_bytes_sent / n):.3f}, recv_avg_gb={_bytes_to_gb(agg_bytes_recv / n):.3f}"
        )

    sizes_line = ", ".join(f"{cid}={len(client_data[cid])}" for cid in range(NUM_CLIENTS))
    print(f"Client data sizes: {sizes_line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient+Frequency layer sharing (async push-based federated learning)")
    parser.add_argument(
        "--model",
        type=str,
        choices=["SimpleCNN", "SimpleCNN10", "VGG11BN", "VGG13BN", "VGG16BN"],
        default=MODEL_NAME,
        help="Model architecture",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Number of logical layers to share per round (overrides --l-percent if both are set). If omitted, defaults to all logical layers.",
    )
    parser.add_argument(
        "--l-percent",
        type=float,
        default=None,
        help="Percentage of total logical layers to share (0-100). Converted to L automatically.",
    )
    parser.add_argument(
        "--x-percent",
        type=float,
        default=LAYER_SELECTION_X_PERCENT,
        help="Within selected L layers, percentage chosen by gradient score (remaining by frequency)",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model
    TOTAL_LOGICAL_LAYERS = _compute_total_logical_layers(MODEL_NAME)
    LAYER_SELECTION_X_PERCENT = float(np.clip(args.x_percent, 0.0, 100.0))

    if args.L is not None:
        resolved_l = int(args.L)
        if args.l_percent is not None:
            print("Both --L and --l-percent provided. Using --L.")
    elif args.l_percent is not None:
        l_pct = float(np.clip(args.l_percent, 0.0, 100.0))
        resolved_l = int(round((l_pct / 100.0) * TOTAL_LOGICAL_LAYERS))
    else:
        resolved_l = int(TOTAL_LOGICAL_LAYERS)
    LAYER_SELECTION_L = min(max(0, resolved_l), TOTAL_LOGICAL_LAYERS)
    main()
