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

BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1
THRESHOLD = 0.6  # Threshold for weight difference between rounds
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
TIMEOUT = 25 # Timeout in seconds for waiting for models
R_PRIME = 100  # Maximum number of rounds
MINIMUM_ROUNDS = 40  # Minimum rounds before checking termination criteria
COUNT_THRESHOLD = 5  # Number of consecutive rounds for weight difference and no crashes

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
                raise ValueError(f"Input file does not contain enough lines for the specified number of faults ({num_faults}).")
            
            for i in range(num_faults):
                id, fr, y = map(int, lines[4+i].split(','))
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
log_filename = f"min40_crash_test_{TIMEOUT}_log_{NUM_CLIENTS}_{NUM_MACHINES}_{len(faults)}.txt"
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
        if message.strip():  # Avoid logging empty lines
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
    num_classes = 10  # CIFAR-10 has 10 classes
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
    
    # Truncate or oversample to ensure fixed_data_per_client for each client
    final_client_indices = {}
    for client_id, indices in client_indices.items():
        np.random.shuffle(indices)
        if len(indices) > fixed_data_per_client:
            final_client_indices[client_id] = indices[:fixed_data_per_client]
        else:
            # Oversample if not enough data
            final_client_indices[client_id] = np.random.choice(
                indices, fixed_data_per_client, replace=True
            ).tolist()

    client_data = [
        torch.utils.data.Subset(dataset, final_client_indices[i]) for i in range(num_clients)
    ]
    return client_data

client_data = create_dirichlet_non_iid_splits_fixed(train_dataset, NUM_CLIENTS, alpha=0.5, fixed_data_per_client=FIXED_DATA_PER_CLIENT)

msg_lck = threading.Lock()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # CIFAR-10 has 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjusted for CIFAR-10 image dimensions
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

def broadcast_weights(id, weights, current_round, terminate, ips, latest_models, crash_away_list, prev_list):
    global model_messages
    message = {'type': 'weights', 'weights': weights, 'round': current_round, 'terminate': terminate, 'id': id}
    for pid in adj[id]:
        with msg_lck:
            model_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)
    latest_models[id] = weights  # Store own model

def broadcast_terminate(id, ips):
    global terminate_messages
    message = {'type': 'terminate'}
    for pid in adj[id]:
        terminate_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)

def tcp_server(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((local_ip, 8650 + id))
    server.listen(NUM_CLIENTS - 1)

    while True:
        conn, addr = server.accept()
        msg = receive_message(conn)
        if msg['type'] == 'terminate':
            terminate_flags.append(1)
            break
        if msg['type'] == 'weights':
            received_weights.append(msg)
            if msg['terminate'] == 1:
                terminate_flags.append(1)
            latest_models[msg['id']] = msg['weights']  # Update latest models
            if msg['id'] not in prev_list[id]:
                prev_list[id].append(msg['id'])
        conn.close()
    server.close()

def average_weights(weights_list):
    avg_weights = []
    for weights_tuple in zip(*weights_list):
        avg_weights.append(np.mean(weights_tuple, axis=0))
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

def client_logic(id, local_ip, ips, faults):
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    previous_weights = None
    current_round = 0
    received_weights = []
    terminate_flags = []
    counter = 0
    crash_counter = 0
    latest_models = defaultdict(dict)
    crash_away_list = [False] * NUM_CLIENTS
    prev_list = [[] for _ in range(NUM_CLIENTS)]
    crashed_in_rounds = []  # Track rounds with new crashes

    # For training time and communication time
    total_training_time = 0.0
    total_comm_time = 0.0

    server_thread = threading.Thread(target=tcp_server, args=(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list))
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
        
        # Check if this client should crash
        for fault in faults:
            if fault[0] == id and fault[1] == current_round:
                for _ in range(fault[2]):
                    broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
                print(f"Client {id} is crashing at round {current_round}")
                return

        # Check if termination flag is received from other clients
        if terminate_flags:
            print(f"Client {id} received termination flag at round {current_round}")
            broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
            break

        broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)

        comm_start = time.time()
        t_start = time.time()
        while (time.time() - t_start) < TIMEOUT:
            time.sleep(0.01)
        comm_end = time.time()
        total_comm_time += (comm_end - comm_start)
        
        # Update crash_away_list and detect new crashes for logging purposes only
        new_crashes = False
        for client_id in range(NUM_CLIENTS):
            if client_id not in [msg['id'] for msg in received_weights] and not crash_away_list[client_id] and client_id != id:
                crash_away_list[client_id] = True
                new_crashes = True
                print(f"Client {id} detected crash of client {client_id} at round {current_round}")

        if new_crashes:
            crashed_in_rounds.append(current_round)
            crash_counter = 0  # Reset counter if new crash detected
        else:
            crash_counter += 1  # Increment if no new crashes

        # Include all received weights, even from crashed clients in previous rounds
        total_weights = [msg['weights'] for msg in received_weights] + [weights]
        new_weights = average_weights(total_weights)
        for param, new_weight in zip(model.parameters(), new_weights):
            param.data = torch.tensor(new_weight).to(DEVICE)

        accuracy = compute_accuracy(model, test_loader)
        print(f"Client {id} - Round {current_round}: Accuracy: {accuracy:.2f}%")

        # Only check for termination criteria after minimum rounds
        if current_round >= MINIMUM_ROUNDS:
            if previous_weights is not None and models_are_similar(new_weights, previous_weights, THRESHOLD):
                counter += 1
            else:
                counter = 0

            # Check if no new crashes in the last COUNT_THRESHOLD rounds
            no_recent_crashes = True
            for r in range(current_round - COUNT_THRESHOLD + 1, current_round + 1):
                if r in crashed_in_rounds:
                    no_recent_crashes = False
                    break

            if counter >= COUNT_THRESHOLD and no_recent_crashes:
                print(f"Client {id} met termination criteria at round {current_round}: stable weights for {COUNT_THRESHOLD} rounds and no crashes")
                broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
                break

        previous_weights = new_weights
        current_round += 1
        received_weights.clear()

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)

    print(f"Client {id} finished. Training time: {total_training_time:.2f}s, Communication time: {total_comm_time:.2f}s, Total: {total_training_time + total_comm_time:.2f}s")
    broadcast_terminate(id, ips)
    server_thread.join()

def main():
    global model_messages, terminate_messages
    start_time = time.time()
    print("Starting Federated Learning on Machine 1 (192.168.50.53)")

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
    print(f"Total model messages passed: {total_model_messages-((NUM_CLIENTS//2)*(NUM_CLIENTS-1))-total_termination_messages}")
    print("Total Termination Messages Passed:", total_termination_messages)
    print(f"Total Time Taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
