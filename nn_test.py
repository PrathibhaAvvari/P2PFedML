import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

# ----------------- Models -----------------

class SimpleCNN10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    raise ValueError("Model choice must be 1..8")


# ----------------- Training -----------------


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def _state_dict_to_list_sorted(model: nn.Module):
    state = model.state_dict()
    return [state[k].detach().cpu().numpy() for k in sorted(state.keys())]


def models_are_similar(weights1, weights2, threshold):
    for w1, w2 in zip(weights1, weights2):
        norm = np.linalg.norm(w1 - w2)
        if norm > threshold:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Single-machine CIFAR-10 training with selectable CNN backbone")
    parser.add_argument(
        "--model",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="1=SimpleCNN10, 2=ResNet18, 3=ResNet34, 4=WideResNet-28-10, 5=DenseNet121, 6=VGG16-BN, 7=VGG11-BN, 8=VGG13-BN"
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-rounds", type=int, default=40)
    parser.add_argument("--count-threshold", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    }
    model_name = model_name_map.get(args.model, "Unknown")

    device = torch.device(args.device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Max epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    best_acc = 0.0
    best_epoch = -1
    previous_weights = None
    stable_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = compute_accuracy(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, test_acc={test_acc:.2f}%")

        current_weights = _state_dict_to_list_sorted(model)
        if epoch >= args.min_rounds:
            if previous_weights is not None and models_are_similar(current_weights, previous_weights, args.threshold):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= args.count_threshold:
                print(
                    f"Early stop: stable weights for {args.count_threshold} epochs (threshold={args.threshold})"
                )
                break

        previous_weights = current_weights

    total_time = time.time() - start_time
    print(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
