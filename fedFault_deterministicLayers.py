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

np.random.seed(42)
torch.manual_seed(42)


# --- Timing instrumentation (per-client) ---
_timing_lock = threading.Lock()
client_timing = defaultdict(lambda: {
    "training_s": 0.0,
    "send_s": 0.0,
    "recv_s": 0.0,
    # Wall-clock time spent in the communication phase (includes socket waiting/timeouts).
    "comm_phase_s": 0.0,
})


def _add_timing(client_id, key, delta_s):
    if client_id is None:
        return
    if delta_s is None or delta_s <= 0:
        return
    with _timing_lock:
        client_timing[client_id][key] += float(delta_s)


BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1
THRESHOLD = 0.6  # Threshold for weight difference between rounds
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
TIMEOUT = 25  # Timeout in seconds for waiting for models
R_PRIME = 200  # Maximum number of rounds
MINIMUM_ROUNDS = 40  # Minimum rounds before checking termination criteria
COUNT_THRESHOLD = 5  # Number of consecutive rounds for weight difference and no crashes

# FIX 1 — Deterministic Layer Sharing Policy
SHARED_LAYERS_ALWAYS = {"conv1", "conv2"}
SHARED_LAYERS_PERIODIC = {"fc1", "fc2"}
PERIODICITY = 3

# FIX 2 — Layer-wise Weighted Averaging
LAYER_ALPHA = {
    "conv1": 0.2,
    "conv2": 0.1,
    "fc1": 0.1,
    "fc2": 0.3,
}


def send_message(conn, message):
    data = pickle.dumps(message)
    message_length = struct.pack('!I', len(data))
    conn.sendall(message_length + data)


def receive_message(conn):
    message_length_data = conn.recv(4)
    if not message_length_data:
        return None
    message_length = struct.unpack('!I', message_length_data)[0]
    data = b''
    while len(data) < message_length:
        part = conn.recv(min(4096, message_length - len(data)))
        data += part
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

# Dynamically name the log file as test_log_<num_clients>_<num_machines>_<num_crashes>.txt
log_filename = f"min40_crash_test_{TIMEOUT}_deterministicLayers_log_{NUM_CLIENTS}_{NUM_MACHINES}_{len(faults)}.txt"
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

retries_list = [1] * NUM_CLIENTS
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


def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.5, fixed_data_per_client=5000):
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

    final_client_indices = {}
    for client_id, indices in client_indices.items():
        np.random.shuffle(indices)
        if len(indices) > fixed_data_per_client:
            final_client_indices[client_id] = indices[:fixed_data_per_client]
        else:
            final_client_indices[client_id] = np.random.choice(indices, fixed_data_per_client, replace=True).tolist()

    client_data = [torch.utils.data.Subset(dataset, final_client_indices[i]) for i in range(num_clients)]
    return client_data


client_data = create_dirichlet_non_iid_splits_fixed(
    train_dataset, NUM_CLIENTS, alpha=0.5, fixed_data_per_client=FIXED_DATA_PER_CLIENT
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
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    retries = 1
    while retries > 0:
        try:
            client.connect((target_ip, 8650 + target_id))
            send_message(client, message)
            client.close()
            break
        except ConnectionRefusedError:
            retries -= 1
            retries_list[target_id] -= 1
            time.sleep(1)
            if retries == 0:
                break


def tcp_client_request_layers(requester_id, target_id, target_ip, param_names, current_round, deadline_ts=None):
    global retries_list
    if not param_names:
        return None

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    retries = 1
    while retries > 0:
        try:
            if deadline_ts is not None:
                timeout_s = max(0.1, float(deadline_ts - time.time()))
                client.settimeout(timeout_s)
            else:
                client.settimeout(5.0)

            _t_send0 = time.perf_counter()
            client.connect((target_ip, 8650 + target_id))
            with msg_lck:
                model_messages[requester_id] += 1

            send_message(
                client,
                {
                    'type': 'layer_request',
                    'requester_id': requester_id,
                    'id': requester_id,
                    'round': current_round,
                    'params': list(param_names),
                },
            )
            _add_timing(requester_id, "send_s", time.perf_counter() - _t_send0)

            _t_recv0 = time.perf_counter()
            resp = receive_message(client)
            _add_timing(requester_id, "recv_s", time.perf_counter() - _t_recv0)
            client.close()
            if resp is None:
                return None
            if resp.get('type') != 'layer_response':
                return None
            if resp.get('round') != current_round:
                return None
            return resp.get('params', None)
        except (ConnectionRefusedError, socket.timeout, OSError):
            retries -= 1
            retries_list[target_id] -= 1
            try:
                client.close()
            except Exception:
                pass
            time.sleep(0.05)
            if retries == 0:
                break
    return None


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


def tcp_server(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list, stop_event):
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
        msg = receive_message(conn)
        _add_timing(id, "recv_s", time.perf_counter() - _t_recv0)
        if msg is None:
            conn.close()
            continue
        if msg['type'] == 'terminate':
            terminate_flags.append(1)
            conn.close()
            break
        if msg['type'] == 'layer_request':
            requested_params = msg.get('params', [])
            round_id = msg.get('round', None)

            with latest_models_lock:
                local_snapshot = latest_models.get(id, None)
                if local_snapshot is None or not isinstance(local_snapshot, dict):
                    local_snapshot = {}

            payload = {}
            for pname in requested_params:
                if pname in local_snapshot:
                    payload[pname] = local_snapshot[pname]

            _t_send0 = time.perf_counter()
            send_message(
                conn,
                {
                    'type': 'layer_response',
                    'provider_id': id,
                    'id': id,
                    'round': round_id,
                    'params': payload,
                },
            )
            _add_timing(id, "send_s", time.perf_counter() - _t_send0)
        conn.close()
    server.close()


def client_logic(id, local_ip, ips, faults):
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    previous_weights_list = None
    current_round = 0
    received_weights = []
    terminate_flags = []
    counter = 0
    crash_counter = 0
    latest_models = defaultdict(dict)
    crash_away_list = [False] * NUM_CLIENTS
    prev_list = [[] for _ in range(NUM_CLIENTS)]
    crashed_in_rounds = []

    stop_event = threading.Event()

    server_thread = threading.Thread(
        target=tcp_server,
        args=(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list, stop_event),
    )
    server_thread.start()
    time.sleep(2)

    conv1_frozen = False

    while current_round < R_PRIME:
        # FIX 3 — Freeze Early Layers After Warm-up
        if (not conv1_frozen) and current_round >= 50:
            for name, param in model.named_parameters():
                if name.startswith("conv1."):
                    param.requires_grad = False
            conv1_frozen = True

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

        # Update what this client can serve to other peers (pull-based protocol).
        with latest_models_lock:
            latest_models[id] = local_state_np

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

        # --- Deterministic layer selection + pull-based layer collection (bandwidth-efficient) ---
        # FIX 1 — Deterministic Layer Sharing Policy
        layers_to_pull = set(SHARED_LAYERS_ALWAYS)
        if (current_round % PERIODICITY) == 0:
            layers_to_pull |= set(SHARED_LAYERS_PERIODIC)

        # Skip known crashed peers in assignment to improve accuracy and reduce wasted timeouts
        neighbors = [n for n in adj[id] if not crash_away_list[n]]
        layer_order = ["conv1", "conv2", "fc1", "fc2"]

        assignment = {}
        for layer_key in layers_to_pull:
            if layer_key not in layer_groups:
                continue
            if not neighbors:
                assignment[layer_key] = id
                continue
            layer_idx = layer_order.index(layer_key) if layer_key in layer_order else 0
            assignment[layer_key] = neighbors[(current_round + layer_idx) % len(neighbors)]

        # Build per-peer request lists (param names) for selected layers only
        params_needed_by_peer = defaultdict(list)
        for layer_key, chosen_peer in assignment.items():
            if chosen_peer == id:
                continue
            params_needed_by_peer[chosen_peer].extend(layer_groups[layer_key])

        deadline_ts = time.time() + TIMEOUT
        pulled_params_by_peer = {}
        responded_peers = set()

        _t_comm0 = time.perf_counter()
        for peer_id, param_names in params_needed_by_peer.items():
            if time.time() >= deadline_ts:
                break
            resp_params = tcp_client_request_layers(
                requester_id=id,
                target_id=peer_id,
                target_ip=ips[peer_id],
                param_names=param_names,
                current_round=current_round,
                deadline_ts=deadline_ts,
            )
            if resp_params is not None:
                pulled_params_by_peer[peer_id] = resp_params
                responded_peers.add(peer_id)
        _add_timing(id, "comm_phase_s", time.perf_counter() - _t_comm0)

        # Crash detection (pull-based): only flag peers we attempted to contact.
        new_crashes = False
        for peer_id in params_needed_by_peer.keys():
            if peer_id not in responded_peers and not crash_away_list[peer_id]:
                crash_away_list[peer_id] = True
                new_crashes = True
                print(f"Client {id} detected crash/unreachable peer {peer_id} at round {current_round}")

        if new_crashes:
            crashed_in_rounds.append(current_round)
            crash_counter = 0
        else:
            crash_counter += 1

        # FIX 2 — Layer-wise Weighted Averaging
        # Apply weighted averaging only to parameters actually pulled from peers.
        new_state_np = dict(local_state_np)
        for layer_key, chosen_peer in assignment.items():
            if chosen_peer == id:
                continue
            resp = pulled_params_by_peer.get(chosen_peer, {})
            for pname in layer_groups.get(layer_key, []):
                if pname not in resp:
                    continue
                alpha = float(LAYER_ALPHA.get(_logical_layer_key(pname), 0.0))
                new_state_np[pname] = (1.0 - alpha) * local_state_np[pname] + alpha * resp[pname]

        # Load back to model
        model.load_state_dict(_numpy_to_state_dict_torch(new_state_np), strict=True)

        accuracy = compute_accuracy(model, test_loader)
        print(f"Client {id} - Round {current_round}: Accuracy: {accuracy:.2f}%")

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
                    latest_models=latest_models,
                    crash_away_list=crash_away_list,
                    prev_list=prev_list,
                )
                break

        previous_weights_list = new_weights_list
        current_round += 1
        received_weights.clear()

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_terminate(id, ips)

    print(f"Client {id} finished.")
    with _timing_lock:
        t = dict(client_timing.get(id, {}))
    train_s = float(t.get("training_s", 0.0))
    send_s = float(t.get("send_s", 0.0))
    recv_s = float(t.get("recv_s", 0.0))
    comm_phase_s = float(t.get("comm_phase_s", 0.0))
    comm_io_s = send_s + recv_s
    print(
        f"Client {id} timing (s): train={train_s:.2f}, comm_io={comm_io_s:.2f} [send {send_s:.2f}, recv {recv_s:.2f}], comm_phase={comm_phase_s:.2f}, comm_total={comm_io_s + comm_phase_s:.2f}, total(train+comm_phase)={train_s + comm_io_s + comm_phase_s:.2f}"
    )
    broadcast_terminate(id, ips)
    stop_event.set()
    server_thread.join()


def main():
    global model_messages, terminate_messages
    start_time = time.time()
    print("Starting Federated Learning (Deterministic Layer Sharing)")

    threads = []
    for i in range(NUM_CLIENTS):
        if ips[i] == str(CURRENT_MACHINE_IP):
            client_thread = threading.Thread(target=client_logic, args=(i, CURRENT_MACHINE_IP, ips, faults))
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

    # Per-client timing summary for clients running on this machine
    local_client_ids = sorted([i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)])
    if local_client_ids:
        with _timing_lock:
            snapshot = {cid: dict(client_timing.get(cid, {})) for cid in local_client_ids}
        print("\nPer-client timing summary (seconds)")
        for cid in local_client_ids:
            t = snapshot.get(cid, {})
            train_s = float(t.get("training_s", 0.0))
            send_s = float(t.get("send_s", 0.0))
            recv_s = float(t.get("recv_s", 0.0))
            comm_phase_s = float(t.get("comm_phase_s", 0.0))
            comm_io_s = send_s + recv_s
            total_s = train_s + comm_io_s + comm_phase_s
            print(
                f"Client {cid}: train={train_s:.2f}, comm_io={comm_io_s:.2f}, comm_phase={comm_phase_s:.2f}, comm_total={comm_io_s + comm_phase_s:.2f}, total(train+comm_phase)={total_s:.2f}"
            )


if __name__ == "__main__":
    main()
