import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import os
import json
import time
import sys

NUM_RUNS = 10
DATA_RATIO = 0.03
STEPS_TEACHER = 15000
STEPS_STUDENT = 15000
RESULTS_FILE = "experiment_results.json"
BATCH_SIZE = 128
DATA_DIR = './data'

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

def get_dataloaders(subset_ratio=1.0, seed=42):
    full_train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    if subset_ratio < 1.0:
        np.random.seed(seed)
        subset_size = int(len(full_train_dataset) * subset_ratio)
        indices = np.random.choice(len(full_train_dataset), subset_size, replace=False)
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, 
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.net(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_cycle(model, teacher, train_loader, test_loader, steps, mode, T=3.0, alpha=0.3):
    model = model.to(device)
    if teacher: teacher.eval()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    step = 0
    model.train()
    
    LOG_INTERVAL = 50
    bar_len = 25

    while step < steps:
        for data, target in train_loader:
            if step >= steps: break
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits_student = model(data)

            if mode == 'distilled' and teacher is not None:
                with torch.no_grad():
                    logits_teacher = teacher(data)
                
                loss_hard = F.cross_entropy(logits_student, target)
                loss_soft = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(logits_student / T, dim=1),
                    F.softmax(logits_teacher / T, dim=1)
                )
                loss = alpha * loss_hard + (1 - alpha) * (T * T) * loss_soft
            else:
                loss = F.cross_entropy(logits_student, target)

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % LOG_INTERVAL == 0 or step == steps:
                percent = step / steps
                filled_len = int(bar_len * percent)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                
                sys.stdout.write(f'\r      [{bar}] {percent*100:.1f}% | Step {step}/{steps} | Loss: {loss.item():.4f}')
                sys.stdout.flush()

    sys.stdout.write('\r')
    sys.stdout.flush()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(1) == target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total

if __name__ == '__main__':
    all_results = []
    
    print(f"Starting Experiment: {NUM_RUNS} runs")
    print(f"Teacher Steps: {STEPS_TEACHER} Student Steps: {STEPS_STUDENT}")
    print(f"Student Data: {DATA_RATIO*100}%")

    start_time = time.time()

    for i in range(1, NUM_RUNS + 1):
        run_seed = 1000 + import
        print(f"RUN {i}/{NUM_RUNS} (Seed: {run_seed})")

        print("Training Teacher")
        loader_full, loader_test = get_dataloaders(1.0, seed=run_seed)
        teacher = TeacherNet()
        acc_teacher = train_cycle(teacher, None, loader_full, loader_test, STEPS_TEACHER, 'teacher')
        print(f"Teacher Acc: {acc_teacher:.2f}%")

        loader_small, _ = get_dataloaders(DATA_RATIO, seed=run_seed)

        print("Baseline Student")
        student_base = StudentNet()
        acc_base = train_cycle(student_base, None, loader_small, loader_test, STEPS_STUDENT, 'baseline')
        print(f"Baseline Acc: {acc_base:.2f}%")

        print("3. Training Distilled Student")
        student_dist = StudentNet()
        acc_dist = train_cycle(student_dist, teacher, loader_small, loader_test, STEPS_STUDENT, 'distilled', T=3.0, alpha=0.3)
        print(f"Distilled Acc: {acc_dist:.2f}%")

        run_data = {
            "run_id": i,
            "seed": run_seed,
            "teacher_acc": acc_teacher,
            "baseline_acc": acc_base,
            "distilled_acc": acc_dist,
            "improvement": acc_dist - acc_base
        }
        all_results.append(run_data)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)
