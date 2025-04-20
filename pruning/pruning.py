'''
Pruning ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import time

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.ff = torch.nn.quantized.FloatFunctional()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ff.add(self.shortcut(x), out)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

import torchvision.models as models
from collections import OrderedDict

def count_parameters():
    model = models.resnet18(pretrained=False)
    conv_layers = OrderedDict()
    linear_layers = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            params = module.weight.numel()
            conv_layers[name] = params
        elif isinstance(module, nn.Linear):
            params = module.weight.numel()
            linear_layers[name] = params
    
    print("Conv:")
    total_conv_params = 0
    for name, params in conv_layers.items():
        print(f"{name}: {params:,} params")
        total_conv_params += params
    print(f"Total conv layers: {len(conv_layers)}")
    print(f"Total conv params: {total_conv_params:,}")
    
    print("\nLinear:")
    total_linear_params = 0
    for name, params in linear_layers.items():
        print(f"{name}: {params:,} params")
        total_linear_params += params
    print(f"Total linear layers: {len(linear_layers)}")
    print(f"Total linear params: {total_linear_params:,}")
    
    return conv_layers, linear_layers

def test(model, dataloader, cuda=False):
    model.eval()
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    inference_time = time.time() - start_time
    
    accuracy = 100 * correct / total
    return accuracy

def prune_layer_by_percentage(model, layer_name, amount=0.9):
    import torch.nn.utils.prune as prune
    import copy
    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
    
    return pruned_model



def layer_by_layer_pruning(model, testloader, device="cpu"):
    import matplotlib.pyplot as plt
    results = {}
    
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prunable_layers.append(name)
    
    baseline_accuracy = test(model, testloader)
    print(f"Baseline acc: {baseline_accuracy:.2f}%")
    
    for layer_name in prunable_layers:
        print(f"Pruning: {layer_name}")
        pruned_model = prune_layer_by_percentage(model, layer_name, amount=0.9)
        accuracy = test(pruned_model, testloader)
        results[layer_name] = accuracy
        print(f"acc {layer_name}: {accuracy:.2f}%")
    
    plt.figure(figsize=(15, 8))
    plt.bar(results.keys(), results.values())
    plt.axhline(y=baseline_accuracy, color='r', linestyle='-', label=f'Baseline ({baseline_accuracy:.2f}%)')
    plt.xticks(rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy after Pruning Individual Layers by 90%')
    plt.tight_layout()
    plt.legend()
    plt.savefig('layer_pruning_results.png')
    plt.close()
    
    return results

def prune_all_linear_layers(model, amount):
    import torch.nn.utils.prune as prune
    import copy
    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount/100.0)
            prune.remove(module, 'weight')
    
    return pruned_model

def binary_search_max_pruning_rate(model, testloader, max_accuracy_drop=2.0, device="cpu"):
    # Get baseline accuracy
    baseline_accuracy = test(model, testloader)
    print(f"Baseline: {baseline_accuracy:.2f}%")
    
    low = 0
    high = 100 
    
    max_k = 0
    
    while low <= high:
        mid = (low + high) // 2
        print(f"current: {mid}%")
        
        pruned_model = prune_all_linear_layers(model, mid)
        accuracy = test(pruned_model, testloader)
        
        accuracy_drop = baseline_accuracy - accuracy
        print(f"{mid}% pruning: {accuracy:.2f}% (drop: {accuracy_drop:.2f}%)")
        
        if accuracy_drop <= max_accuracy_drop:
            max_k = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(f"max pruning rate: {max_k}%")
    
    final_model = prune_all_linear_layers(model, max_k)
    final_accuracy = test(final_model, testloader)
    final_drop = baseline_accuracy - final_accuracy
    
    print(f"Q3: {max_k}% pruning: {final_accuracy:.2f}% (drop: {final_drop:.2f}%)")
    
    return max_k


if __name__ == "__main__":
    
    print("---------------CIFAR10 dataset---------------")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)
    
    print("========= TASK 1 =========")
    device = "cpu"
    model = ResNet18()
    model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
    model.eval()
    
    count_parameters()
    
    # print("========= TASK 2 =========")
    
    layer_by_layer_pruning(model, testloader, device)
    
    print("========= TASK 3 =========")
    max_k = binary_search_max_pruning_rate(model, testloader, max_accuracy_drop=2.0, device=device)
    