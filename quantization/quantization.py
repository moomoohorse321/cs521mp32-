'''ResNet in PyTorch.

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
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

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
    def __init__(self, block, num_blocks, num_classes=10, q=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.q = q

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.q:
            out = self.dequant(out)
        return out


def ResNet18(q=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], q=q)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print(f'Size (MB): {size:.2f}')
    os.remove('temp.p')
    return size

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
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Inference Time: {inference_time:.2f} secs')
    
    return accuracy, inference_time


def main():
    device = torch.device("cpu")
    
    print("---------------CIFAR10 dataset---------------")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)
    
    calibration_set = torch.utils.data.Subset(testset, indices=list(range(1000)))
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=128,
                                                   shuffle=False, num_workers=2)
    
    # T1: original size and accuracy
    print("========= TASK 1 =========")
    model_fp32 = ResNet18(q=False)
    model_fp32.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
    
    print("Original FP32 Model Size:")
    original_size = print_size_of_model(model_fp32)
    
    original_accuracy, original_time = test(model_fp32, testloader, False)
    
    
    # T2: Quantization
    print("========= TASK 2=========")
    model_to_quantize = ResNet18(q=True)
    model_to_quantize.load_state_dict(model_fp32.state_dict())
    model_to_quantize.eval() 
    
    from torch.quantization.observer import MovingAverageMinMaxObserver
    model_to_quantize.qconfig = torch.quantization.QConfig(
                                      activation=MovingAverageMinMaxObserver.with_args(reduce_range=True), 
                                      weight=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    print(f"Quantization config: {model_to_quantize.qconfig}")
    
    torch.quantization.prepare(model_to_quantize, inplace=True)
    
    print("Calibrating...")
    test(model_to_quantize, calibration_loader, cuda=False)
    
    print("Converting...")
    model_int8 = torch.quantization.convert(model_to_quantize, inplace=True)
    
    print("Quantized INT8 Model Size:")
    quantized_size = print_size_of_model(model_int8)
    
    quantized_accuracy, quantized_time = test(model_int8, testloader, cuda=False)
    
    # T3: inference time
    print("========= TASK 3=========")
    print("Re-measuring inference times...")
    
    num_iterations = 3
    fp32_times = []
    int8_times = []
    
    for i in range(num_iterations):
        print("Testing FP32 model...")
        _, fp32_time = test(model_fp32, testloader, cuda=False)
        fp32_times.append(fp32_time)
        
        print("Testing INT8 model...")
        _, int8_time = test(model_int8, testloader, cuda=False)
        int8_times.append(int8_time)
    
    avg_fp32_time = sum(fp32_times) / num_iterations
    avg_int8_time = sum(int8_times) / num_iterations
    
    # Print summary of results
    print("\n========= SUMMARY =========")
    print(f"Q1: Original Size: {original_size:.2f} MB")
    print(f"Original Accuracy: {original_accuracy:.2f}%")
    
    print(f"Q2: Quantized Size: {quantized_size:.2f} MB")
    print(f"Q2: Quantized Accuracy: {quantized_accuracy:.2f}%")
    print(f"Q2: Size Reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
    
    print(f"Q3: Original Average Inference Time: {avg_fp32_time:.2f} seconds")
    print(f"Q3: Quantized Average Inference Time: {avg_int8_time:.2f} seconds")
    print(f"Q3: Speedup: {avg_fp32_time/avg_int8_time:.2f}x")

if __name__ == '__main__':
    main()