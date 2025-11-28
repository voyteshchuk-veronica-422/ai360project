import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BATCH_SIZE = 128
TEACHER_EPOCHS = 10
STUDENT_EPOCHS = 10
NUM_RUNS = 3

class FashionTeacher(nn.Module):
    def __init__(self):
        super(FashionTeacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class Student(nn.Module):
    def __init__(self, input_size=784):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x

def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST("data/", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("data/", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / total

def train_teacher_model():
    print("\n--- Training Teacher ---")
    train_loader, test_loader = get_dataloaders()
    model = FashionTeacher().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{TEACHER_EPOCHS}, teacher accuracy {acc:.2f}%")

    path = "teacher_fashion.pth"
    torch.save(model.state_dict(), path)
    return path

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    
    soft_loss = -(soft_targets * student_log_probs).sum(dim=1).mean()

    loss = alpha * hard_loss + (1.0 - alpha) * (T**2) * soft_loss
    return loss

def train_student(teacher_path, T, alpha):
    train_loader, test_loader = get_dataloaders()
    
    teacher = None
    if teacher_path:
        teacher = FashionTeacher().to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.eval()

    student = Student().to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            student_logits = student(x)

            if teacher is None:
                loss = F.cross_entropy(student_logits, y)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss = distillation_loss(student_logits, teacher_logits, y, T, alpha)

            loss.backward()
            optimizer.step()

    return evaluate(student, test_loader)

if __name__ == "__main__":
    if not os.path.exists("teacher_fashion.pth"):
        teacher_path = train_teacher_model()
    else:
        teacher_path = "teacher_fashion.pth"
        
    t_model = FashionTeacher().to(device)
    t_model.load_state_dict(torch.load(teacher_path, map_location=device))
    _, t_loader = get_dataloaders()
    teacher_acc = evaluate(t_model, t_loader)
    print(f"Teacher accuracy: {teacher_acc:.2f}%")

    results = []

    print("\nBaseline Student")
    baseline_accs = []
    for i in range(NUM_RUNS):
        acc = train_student(None, T=1, alpha=1.0)
        baseline_accs.append(acc)
        print(f"Run {i+1}: {acc:.2f}%")
    
    base_mean = np.mean(baseline_accs)
    base_std = np.std(baseline_accs)
    results.append(["Baseline", "-", "-", f"{base_mean:.2f} ± {base_std:.2f}"])

    T_list = [3, 6]
    Alpha_list = [0.3, 0.5]

    for T in T_list:
        for alpha in Alpha_list:
            print(f"\nDistillation (T={T}, Alpha={alpha})")
            run_accs = []
            for i in range(NUM_RUNS):
                acc = train_student(teacher_path, T, alpha)
                run_accs.append(acc)
                print(f"Run {i+1}: {acc:.2f}%")
            
            mean = np.mean(run_accs)
            std = np.std(run_accs)
            results.append([f"Distilled", T, alpha, f"{mean:.2f} plus-minus {std:.2f}"])

    print(f"{'Model Type':<15} {'T':<5} {'Alpha':<5} {'Accuracy':<20}")
    for row in results:
        print(f"{row[0]:<15} {str(row[1]):<5} {str(row[2]):<5} {row[3]:<20}")
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")