import copy
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---- Config: centralized synchronous FL, 12 clients, no crashes, non-IID alpha=0.5 ----
SEED = 42
NUM_CLIENTS = 12
ALPHA = 0.5
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.02
MODEL_NAME = "SimpleCNN10"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bytes_to_gb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 ** 3)


def _model_state_size_bytes(state_dict) -> int:
    return int(sum(v.numel() * v.element_size() for v in state_dict.values()))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 32 -> 16
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 16 -> 8
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(model_name):
    if model_name == "SimpleCNN10":
        return SimpleCNN10()
    if model_name == "SimpleCNN":
        return SimpleCNN()
    raise ValueError(f"Unknown model: {model_name}")


def dirichlet_partition(labels, num_clients=12, alpha=0.5, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(class_idx)).astype(int)
        counts[-1] = len(class_idx) - counts[:-1].sum()

        start = 0
        for client_id, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[client_id].extend(class_idx[start:start + cnt].tolist())
            start += cnt

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def local_train(model, loader, device, lr=0.01, local_epochs=1):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fedavg(state_dicts):
    avg_state = OrderedDict()
    for k in state_dicts[0].keys():
        tensors = [sd[k] for sd in state_dicts]
        if tensors[0].dtype.is_floating_point:
            avg_state[k] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            avg_state[k] = tensors[0]
    return avg_state


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    client_splits = dirichlet_partition(
        labels=train_dataset.targets,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        num_classes=10,
        seed=SEED,
    )

    client_loaders = []
    print("\nClient data sizes:")
    for cid, idxs in enumerate(client_splits):
        print(f"Client {cid:02d}: {len(idxs)} samples")
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    global_model = get_model(MODEL_NAME).to(device)
    model_size_bytes = _model_state_size_bytes(global_model.state_dict())
    final_local_states = [None] * NUM_CLIENTS
    timing_store = {
        cid: {
            "training_s": 0.0,
            "comm_io_s": 0.0,
            "send_s": 0.0,
            "recv_s": 0.0,
            "comm_phase_s": 0.0,
            "wait_s": 0.0,
            "total_s": 0.0,
            "bytes_sent": 0,
            "bytes_recv": 0,
            "messages_sent": 0,
            "messages_recv": 0,
            "final_acc": 0.0,
            "best_acc": -1.0,
            "best_round": -1,
            "last_round": 0,
            "local_final_acc": 0.0,
        }
        for cid in range(NUM_CLIENTS)
    }
    best_acc = -1.0
    best_round = -1
    final_acc = 0.0

    print("\nStarting centralized synchronous FedAvg...")
    print(f"Clients: {NUM_CLIENTS}, alpha: {ALPHA}, crashes: 0 (all clients always participate)\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        local_states = []

        # Synchronous: server waits for all 12 clients every round
        for cid in range(NUM_CLIENTS):
            timing_store[cid]["bytes_recv"] += model_size_bytes
            timing_store[cid]["messages_recv"] += 1
            local_model = copy.deepcopy(global_model)
            t_train0 = time.perf_counter()
            local_state = local_train(
                model=local_model,
                loader=client_loaders[cid],
                device=device,
                lr=LR,
                local_epochs=LOCAL_EPOCHS,
            )
            timing_store[cid]["training_s"] += time.perf_counter() - t_train0
            timing_store[cid]["bytes_sent"] += model_size_bytes
            timing_store[cid]["messages_sent"] += 1
            local_states.append(local_state)
            if rnd == NUM_ROUNDS:
                final_local_states[cid] = {k: v.detach().cpu() for k, v in local_state.items()}

        t_agg0 = time.perf_counter()
        new_global_state = fedavg(local_states)
        global_model.load_state_dict(new_global_state)
        comm_phase_s = time.perf_counter() - t_agg0
        per_client_comm_phase = comm_phase_s / max(1, NUM_CLIENTS)
        for cid in range(NUM_CLIENTS):
            timing_store[cid]["comm_phase_s"] += per_client_comm_phase

        test_loss, test_acc = evaluate(global_model, test_loader, device)
        final_acc = float(test_acc)
        if test_acc > best_acc:
            best_acc = float(test_acc)
            best_round = int(rnd)
        print(f"Round {rnd:02d}/{NUM_ROUNDS} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    for cid in range(NUM_CLIENTS):
        t = timing_store[cid]
        t["comm_io_s"] = float(t.get("send_s", 0.0)) + float(t.get("recv_s", 0.0))
        t["total_s"] = float(t.get("training_s", 0.0)) + t["comm_io_s"] + float(t.get("comm_phase_s", 0.0)) + float(t.get("wait_s", 0.0))
        t["final_acc"] = final_acc
        t["best_acc"] = best_acc
        t["best_round"] = best_round
        t["last_round"] = NUM_ROUNDS

    for cid in range(NUM_CLIENTS):
        state = final_local_states[cid]
        if state is None:
            continue
        local_model = get_model(MODEL_NAME).to(device)
        local_model.load_state_dict(state)
        _, local_acc = evaluate(local_model, test_loader, device)
        timing_store[cid]["local_final_acc"] = float(local_acc)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nFederated Learning Completed")
    print("Number of Clients:", NUM_CLIENTS)
    print(f"Total Rounds: {NUM_ROUNDS}")
    print(f"Total Time Taken: {total_time:.2f} seconds")

    print("\nGlobal model accuracy summary")
    print(f"  final_acc={final_acc:.2f}%, best_acc={best_acc:.2f}% (round {best_round})")

    print("\nPer-client timing summary (seconds)")
    for cid in range(NUM_CLIENTS):
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
            f"comm_total={comm_io_s + comm_phase_s:.2f}, total={total_s:.2f}, "
            f"bytes_sent={bytes_sent} ({_bytes_to_gb(bytes_sent):.3f} GB), "
            f"bytes_recv={bytes_recv} ({_bytes_to_gb(bytes_recv):.3f} GB)"
        )

    print("\nPer-client accuracy summary (local models on global test set)")
    for cid in range(NUM_CLIENTS):
        t = timing_store.get(cid, {})
        last_rnd = int(t.get("last_round", -1))
        local_final_acc = float(t.get("local_final_acc", 0.0))
        print(
            f"  Client {cid}: local_final_acc={local_final_acc:.2f}%, "
            f"last_round={last_rnd}"
        )

    avg_final_acc = (
        sum(float(timing_store.get(cid, {}).get("local_final_acc", 0.0)) for cid in range(NUM_CLIENTS))
        / max(1, NUM_CLIENTS)
    )
    print(f"\nAverage final local accuracy (all clients): {avg_final_acc:.2f}%")

    agg_train = sum(float(timing_store.get(cid, {}).get("training_s", 0.0)) for cid in range(NUM_CLIENTS))
    agg_comm_io = sum(float(timing_store.get(cid, {}).get("comm_io_s", 0.0)) for cid in range(NUM_CLIENTS))
    agg_comm_phase = sum(float(timing_store.get(cid, {}).get("comm_phase_s", 0.0)) for cid in range(NUM_CLIENTS))
    agg_wait = sum(float(timing_store.get(cid, {}).get("wait_s", 0.0)) for cid in range(NUM_CLIENTS))
    agg_total = sum(float(timing_store.get(cid, {}).get("total_s", 0.0)) for cid in range(NUM_CLIENTS))
    agg_bytes_sent = sum(int(timing_store.get(cid, {}).get("bytes_sent", 0)) for cid in range(NUM_CLIENTS))
    agg_bytes_recv = sum(int(timing_store.get(cid, {}).get("bytes_recv", 0)) for cid in range(NUM_CLIENTS))
    n = max(1, NUM_CLIENTS)
    print("\nAggregate timing (all clients)")
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

    print("\nDone.")


if __name__ == "__main__":
    main()