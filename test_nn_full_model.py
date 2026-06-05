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
import os

np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1
THRESHOLD = 0.6
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
# For deep models, payloads can be very large; allow overriding without code edits.
TIMEOUT = int(os.environ.get("FED_TIMEOUT", "25"))
CONNECT_TIMEOUT = float(os.environ.get("FED_CONNECT_TIMEOUT", "60"))
TCP_RETRIES = int(os.environ.get("FED_TCP_RETRIES", "3"))
SERVER_BACKLOG = int(os.environ.get("FED_SERVER_BACKLOG", "128"))
R_PRIME = 100
MINIMUM_ROUNDS = 40
COUNT_THRESHOLD = 5

# ----------------- Utils -----------------

def send_message(conn, message):
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    message_length = struct.pack('!I', len(data))
    conn.sendall(message_length + data)


def _recv_exact(conn, nbytes: int):
    data = b''
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive_message(conn):
    message_length_data = _recv_exact(conn, 4)
    if not message_length_data:
        return None
    message_length = struct.unpack('!I', message_length_data)[0]
    data = _recv_exact(conn, message_length)
    if data is None:
        return None
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


# ----------------- Models -----------------

# Simple 10-Layer CNN
class SimpleCNN10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            
            # Block 2: 128 channels
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            
            # Block 3: 256 channels
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),
            
            # Block 4: 512 channels
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
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


# VGG variants with BN (CIFAR-friendly)
def _make_vgg_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
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


# BasicBlock for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# WideResNet-28-10 (CIFAR)
class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(self.dropout(torch.relu(self.bn2(out))))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super().__init__()
        assert ((depth - 4) % 6 == 0), "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[0], nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(nStages[1], nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(nStages[2], nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, dropout_rate, stride):
        layers = []
        layers.append(WideBasic(in_planes, planes, dropout_rate, stride))
        for _ in range(num_blocks - 1):
            layers.append(WideBasic(planes, planes, dropout_rate, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn1(out))
        out = torch.mean(out, dim=[2, 3])
        out = self.fc(out)
        return out


def WRN28_10():
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3)


# DenseNet-121 (CIFAR-friendly)
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        blocks = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2
        self.features_blocks = nn.Sequential(*blocks)

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.features_blocks(x)
        x = torch.relu(self.final_bn(x))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def DenseNet121():
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)


# ResNet-20 (CIFAR-specific, He et al. 2015)
# 3 stages [16, 32, 64], no MaxPool on stem, GlobalAvgPool at end.
# n=3 → 6n+2 = 20 layers, ~0.27M params.
class _CifarBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class _CifarResNet(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = [_CifarBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(_CifarBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)


def ResNet20():
    return _CifarResNet(n=3)  # 6*3+2 = 20 layers, ~0.27M params


def build_model(choice: int) -> nn.Module:
    if choice == 1:
        return SimpleCNN10()
    if choice == 2:
        return ResNet18()
    if choice == 3:
        return ResNet34()
    if choice == 4:
        return WRN28_10()
    if choice == 5:
        return DenseNet121()
    if choice == 6:
        return VGG16BN()
    if choice == 7:
        return VGG11BN()
    if choice == 8:
        return VGG13BN()
    if choice == 9:
        return ResNet20()
    raise ValueError("Model choice must be 1..9")


# ----------------- Federated Logic -----------------

NUM_CLIENTS, NUM_MACHINES, CURRENT_MACHINE_IP, ips, faults = parse_input_file()

if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    exit(1)

logger = logging.getLogger('federated_learning')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_filename = f"min40_crash_test_{TIMEOUT}_log_{NUM_CLIENTS}_{NUM_MACHINES}_{len(faults)}.txt"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

class CrashFilter(logging.Filter):
    def filter(self, record):
        return "crash" in record.msg.lower() or "crashing" in record.msg.lower()

file_handler.addFilter(CrashFilter())
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

sys.stdout = LoggerWriter(logger, logging.INFO)

retries_list = [1] * NUM_CLIENTS
adj = [[j for j in range(NUM_CLIENTS) if j != i] for i in range(NUM_CLIENTS)]
terminate_messages = [0] * NUM_CLIENTS
model_messages = [0] * NUM_CLIENTS

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indices = np.random.permutation(len(train_dataset))


def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.1, fixed_data_per_client=5000):
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
    train_dataset, NUM_CLIENTS, alpha=0.1, fixed_data_per_client=FIXED_DATA_PER_CLIENT
)

msg_lck = threading.Lock()


def tcp_client(id, target_id, target_ip, message):
    global retries_list
    retries = TCP_RETRIES
    while retries > 0:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(CONNECT_TIMEOUT)
        try:
            # Bump buffers for large payloads.
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except Exception:
            pass
        try:
            client.connect((target_ip, 8650 + target_id))
            send_message(client, message)
            return True
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError, socket.timeout):
            retries -= 1
            retries_list[target_id] -= 1
            time.sleep(1)
        finally:
            try:
                client.close()
            except Exception:
                pass
    return False


def broadcast_weights(id, weights, current_round, terminate, ips, latest_models, crash_away_list, prev_list):
    global model_messages
    message = {'type': 'weights', 'weights': weights, 'round': current_round, 'terminate': terminate, 'id': id}
    for pid in adj[id]:
        with msg_lck:
            model_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)
    latest_models[id] = weights


def broadcast_terminate(id, ips):
    global terminate_messages
    message = {'type': 'terminate'}
    for pid in adj[id]:
        terminate_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)


def tcp_server(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list, received_lock):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind to all interfaces so misconfigured local_ip doesn't break multi-machine runs.
    server.bind(("0.0.0.0", 8650 + id))
    server.listen(SERVER_BACKLOG)

    stop_event = threading.Event()

    def handle_conn(conn):
        try:
            try:
                conn.settimeout(CONNECT_TIMEOUT)
            except Exception:
                pass
            msg = receive_message(conn)
            if not msg:
                return
            with received_lock:
                if msg.get('type') == 'terminate':
                    terminate_flags.append(1)
                    stop_event.set()
                    return
                if msg.get('type') == 'weights':
                    received_weights.append(msg)
                    if msg.get('terminate') == 1:
                        terminate_flags.append(1)
                    latest_models[msg['id']] = msg['weights']
                    if msg['id'] not in prev_list[id]:
                        prev_list[id].append(msg['id'])
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        while not stop_event.is_set():
            conn, _addr = server.accept()
            threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()
    finally:
        try:
            server.close()
        except Exception:
            pass


def average_weights(weights_list):
    avg_weights = []
    for weights_tuple in zip(*weights_list):
        stacked = np.stack([w.astype(np.float32, copy=False) for w in weights_tuple], axis=0)
        avg_weights.append(stacked.mean(axis=0))
    return avg_weights


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


def models_are_similar(weights1, weights2, threshold):
    for w1, w2 in zip(weights1, weights2):
        norm = np.linalg.norm(w1 - w2)
        if norm > threshold:
            return False
    return True


def client_logic(id, local_ip, ips, faults, model_choice, timing_store):
    model = build_model(model_choice).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    previous_weights = None
    current_round = 0
    received_weights = []
    received_lock = threading.Lock()
    terminate_flags = []
    counter = 0
    crash_counter = 0
    latest_models = defaultdict(dict)
    crash_away_list = [False] * NUM_CLIENTS
    prev_list = [[] for _ in range(NUM_CLIENTS)]
    crashed_in_rounds = []

    total_training_time = 0.0
    total_comm_time = 0.0
    best_accuracy = 0.0
    best_round = -1
    final_accuracy = 0.0

    server_thread = threading.Thread(
        target=tcp_server,
        args=(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list, received_lock)
    )
    server_thread.start()
    time.sleep(2)

    while current_round < R_PRIME:
        train_start = time.time()
        model.train()
        for epoch in range(EPOCHS_PER_ROUND):
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        train_end = time.time()
        total_training_time += (train_end - train_start)

        weights = [param.cpu().detach().numpy() for param in model.parameters()]

        for fault in faults:
            if fault[0] == id and fault[1] == current_round:
                for _ in range(fault[2]):
                    broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models,
                                     crash_away_list=crash_away_list, prev_list=prev_list)
                print(f"Client {id} is crashing at round {current_round}")
                return

        if terminate_flags:
            print(f"Client {id} received termination flag at round {current_round}")
            broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models,
                             crash_away_list=crash_away_list, prev_list=prev_list)
            break

        broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models,
                         crash_away_list=crash_away_list, prev_list=prev_list)

        comm_start = time.time()
        t_start = time.time()
        # Wait for current round messages; for large models, fixed sleep can be too short.
        while (time.time() - t_start) < TIMEOUT:
            with received_lock:
                got_ids = {m['id'] for m in received_weights if m.get('type') == 'weights' and m.get('round') == current_round}
            if all((pid in got_ids) or crash_away_list[pid] for pid in adj[id]):
                break
            time.sleep(0.05)
        comm_end = time.time()
        total_comm_time += (comm_end - comm_start)

        with received_lock:
            current_round_msgs = [m for m in received_weights if m.get('type') == 'weights' and m.get('round') == current_round]

        new_crashes = False
        for client_id in range(NUM_CLIENTS):
            if client_id != id and (client_id not in [msg['id'] for msg in current_round_msgs]) and not crash_away_list[client_id]:
                crash_away_list[client_id] = True
                new_crashes = True
                print(f"Client {id} detected crash of client {client_id} at round {current_round}")

        if new_crashes:
            crashed_in_rounds.append(current_round)
            crash_counter = 0
        else:
            crash_counter += 1

        total_weights = [msg['weights'] for msg in current_round_msgs] + [weights]
        new_weights = average_weights(total_weights)
        for param, new_weight in zip(model.parameters(), new_weights):
            param.data = torch.from_numpy(new_weight).to(device=DEVICE, dtype=param.data.dtype)

        accuracy = compute_accuracy(model, test_loader)
        final_accuracy = float(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_round = int(current_round)
        print(f"Client {id} - Round {current_round}: Accuracy: {accuracy:.2f}%")

        if current_round >= MINIMUM_ROUNDS:
            if previous_weights is not None and models_are_similar(new_weights, previous_weights, THRESHOLD):
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
                    f"Client {id} met termination criteria at round {current_round}: stable weights for {COUNT_THRESHOLD} rounds and no crashes"
                )
                broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models,
                                 crash_away_list=crash_away_list, prev_list=prev_list)
                break

        previous_weights = new_weights
        current_round += 1
        # Drop old rounds; keep future-round messages that arrived early.
        with received_lock:
            received_weights[:] = [m for m in received_weights if m.get('type') == 'weights' and m.get('round', -1) >= current_round]

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models,
                         crash_away_list=crash_away_list, prev_list=prev_list)

    total_time = total_training_time + total_comm_time
    print(
        f"Client {id} finished. Training time: {total_training_time:.2f}s, Communication time: {total_comm_time:.2f}s, Total: {total_time:.2f}s"
    )
    timing_store[id] = {
        "training_s": float(total_training_time),
        "comm_s": float(total_comm_time),
        "total_s": float(total_time),
        "final_acc": float(final_accuracy),
        "best_acc": float(best_accuracy),
        "best_round": int(best_round),
        "last_round": int(current_round),
    }
    broadcast_terminate(id, ips)
    server_thread.join()


def main():
    parser = argparse.ArgumentParser(description="Federated Learning with selectable CNN backbone")
    parser.add_argument(
        "--model",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="1=SimpleCNN10, 2=ResNet18, 3=ResNet34, 4=WideResNet-28-10, 5=DenseNet121, 6=VGG16-BN, 7=VGG11-BN, 8=VGG13-BN, 9=ResNet20-CIFAR"
    )
    args = parser.parse_args()

    model_name_map = {
        1: "SimpleCNN-10",
        2: "ResNet-18",
        3: "ResNet-34",
        4: "WideResNet-28-10",
        5: "DenseNet-121",
        6: "VGG-16-BN",
        7: "VGG-11-BN",
        8: "VGG-13-BN",
        9: "ResNet-20-CIFAR",
    }
    model_name = model_name_map.get(args.model, "Unknown")

    global model_messages, terminate_messages
    start_time = time.time()
    print(f"Starting Federated Learning with model choice {args.model} ({model_name})")

    threads = []
    timing_store = {}
    for i in range(NUM_CLIENTS):
        if ips[i] == str(CURRENT_MACHINE_IP):
            client_thread = threading.Thread(
                target=client_logic,
                args=(i, CURRENT_MACHINE_IP, ips, faults, args.model, timing_store)
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
    print("Model Used:", model_name)
    print(f"Total model messages passed: {total_model_messages-((NUM_CLIENTS//2)*(NUM_CLIENTS-1))-total_termination_messages}")
    print("Total Termination Messages Passed:", total_termination_messages)
    print(f"Total Time Taken: {total_time:.2f} seconds")

    local_client_ids = sorted([i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)])
    if local_client_ids:
        print("\nPer-client timing summary (seconds)")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            train_s = float(t.get("training_s", 0.0))
            comm_s = float(t.get("comm_s", 0.0))
            total_s = float(t.get("total_s", train_s + comm_s))
            print(f"Client {cid}: train={train_s:.2f}, comm={comm_s:.2f}, total={total_s:.2f}")

        print("\nPer-client accuracy summary")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            final_acc = float(t.get("final_acc", 0.0))
            best_acc = float(t.get("best_acc", 0.0))
            best_round = int(t.get("best_round", -1))
            last_round = int(t.get("last_round", -1))
            print(
                f"Client {cid}: final_acc={final_acc:.2f}%, best_acc={best_acc:.2f}% (round {best_round}), last_round={last_round}"
            )

        avg_final_acc = (
            sum(float(timing_store.get(cid, {}).get("final_acc", 0.0)) for cid in local_client_ids)
            / max(1, len(local_client_ids))
        )
        print(f"\nAverage final accuracy (local clients): {avg_final_acc:.2f}%")

        agg_train = sum(float(timing_store.get(cid, {}).get("training_s", 0.0)) for cid in local_client_ids)
        agg_comm = sum(float(timing_store.get(cid, {}).get("comm_s", 0.0)) for cid in local_client_ids)
        agg_total = sum(float(timing_store.get(cid, {}).get("total_s", 0.0)) for cid in local_client_ids)
        print("\nAggregate timing (local clients)")
        print(f"Sum train={agg_train:.2f}, sum comm={agg_comm:.2f}, sum total={agg_total:.2f}")


if __name__ == "__main__":
    main()
