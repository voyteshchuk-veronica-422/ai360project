import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

class MnistNetworkTeacher(nn.Module):
    def __init__(self):
        super(MnistNetworkTeacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, input_tensor, keep_prob_conv, keep_prob_hidden):
        x = F.relu(self.conv1(input_tensor))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=1 - keep_prob_conv, training=self.training)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=1 - keep_prob_conv, training=self.training)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(x, p=1 - keep_prob_conv, training=self.training)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=1 - keep_prob_hidden, training=self.training)
        x = self.fc2(x)
        return x

class MnistNetworkStudent(nn.Module):
    def __init__(self):
        super(MnistNetworkStudent, self).__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, input_tensor):
        x = torch.flatten(input_tensor, 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def loss_and_accuracy(prediction, target):
    cross_entropy = torch.mean(-torch.sum(target * torch.log(prediction + 1e-12), dim=1))
    correct_prediction = torch.eq(torch.argmax(prediction, dim=1), torch.argmax(target, dim=1))
    accuracy = torch.mean(correct_prediction.float())
    return cross_entropy, accuracy

def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes, device=labels.device)[labels]

def calculate_test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if isinstance(model, MnistNetworkTeacher):
                logits = model(batch_x, 1.0, 1.0)
            else:
                logits = model(batch_x)
                
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    return correct / total

def plot_single_model_history(history, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history['steps'], history['train_accuracies'], label='Train Accuracy')
    plt.plot(history['steps'], history['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.02)
    plt.tight_layout()
    filename = model_name.replace(" ", "_").lower() + "_accuracy_history.png"
    plt.savefig(filename)
    print(f"Accuracy graph for {model_name} saved as {filename}")
    plt.close()

def plot_student_comparison(histories):
    colors = ['orange', 'green', 'red']
    
    plt.figure(figsize=(10, 7))
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history['steps'], history['train_accuracies'], label=f'{name} (Train)', color=colors[i], linestyle='-')
    plt.gca().set_prop_cycle(None)
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history['steps'], history['test_accuracies'], label=f'{name} (Test)', color=colors[i], linestyle='--')

    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Student Models: Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.02)
    
    plt.tight_layout()
    filename = "student_models_accuracy_comparison.png"
    plt.savefig(filename)
    print(f"Comparison graph for student models saved as {filename}")
    plt.close()

def train_teacher():
    print("\n--- Teacher Training ---")
    start_lr, decay, steps_total, verbose_step = 1e-4, 1e-6, 15000, 500
    MODEL_SAVE_PATH = './models/teacher1.ckpt'

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST("MNIST_data/", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("MNIST_data/", train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    teacher = MnistNetworkTeacher().to(device)
    optimizer = torch.optim.RMSprop(teacher.parameters(), lr=start_lr, weight_decay=decay)

    history = {'steps': [], 'train_accuracies': [], 'test_accuracies': []}

    teacher.train()
    step = 0
    done = False
    while not done:
        for batch_x, batch_y in train_loader:
            if step >= steps_total: done = True; break
            
            teacher.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_one_hot = to_one_hot(batch_y)

            optimizer.zero_grad()
            logits_teacher = teacher(batch_x, keep_prob_conv=0.8, keep_prob_hidden=0.5)
            y_conv_teacher = F.softmax(logits_teacher, dim=1)
            loss, acc = loss_and_accuracy(y_conv_teacher, y_one_hot)
            loss.backward()
            optimizer.step()

            if (step + 1) % verbose_step == 0:
                test_acc_val = calculate_test_accuracy(teacher, test_loader)
                
                history['steps'].append(step + 1)
                history['train_accuracies'].append(acc.item())
                history['test_accuracies'].append(test_acc_val)
                
                print(f"Teacher | Step {step+1}/{steps_total} | Train Acc: {acc.item():.4f} | Test Acc: {test_acc_val:.4f}")
            
            step += 1
            
    print("--- Teacher Training Finished ---")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(teacher.state_dict(), MODEL_SAVE_PATH)
    print(f"Teacher model saved to {MODEL_SAVE_PATH}")
    return history

def train_student_baseline():
    print("\n--- Baseline Student Training ---")
    steps_total, verbose_step = 15000, 500

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST("MNIST_data/", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("MNIST_data/", train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    student = MnistNetworkStudent().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    history = {'steps': [], 'train_accuracies': [], 'test_accuracies': []}

    student.train()
    step = 0
    done = False
    while not done:
        for batch_x, batch_y in train_loader:
            if step >= steps_total: done = True; break
            
            student.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_one_hot = to_one_hot(batch_y)
            
            optimizer.zero_grad()
            logits_student = student(batch_x)
            y_student_hard = F.softmax(logits_student, dim=1)
            loss, acc = loss_and_accuracy(y_student_hard, y_one_hot)
            loss.backward()
            optimizer.step()

            if (step + 1) % verbose_step == 0:
                test_acc_val = calculate_test_accuracy(student, test_loader)
                
                history['steps'].append(step + 1)
                history['train_accuracies'].append(acc.item())
                history['test_accuracies'].append(test_acc_val)
                
                print(f"Baseline Student | Step {step+1}/{steps_total} | Train Acc: {acc.item():.4f} | Test Acc: {test_acc_val:.4f}")

            step += 1
            
    print("--- Baseline Student Training Finished ---")
    return student, history

def train_student_distilled(alpha: float, temperature: int):
    title = f"Distilled Student: alpha={alpha}, temp={temperature}"
    print(f"\n--- {title} ---")
    
    steps_total, verbose_step = 15000, 500
    TEACHER_MODEL_PATH, STUDENT_SAVE_PATH = './models/teacher1.ckpt', f'./models/student_a{alpha}_t{temperature}.ckpt'
    
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST("MNIST_data/", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("MNIST_data/", train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    teacher = MnistNetworkTeacher().to(device)
    try:
        teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device))
        teacher.eval()
        print("Teacher model loaded.")
    except FileNotFoundError:
        print(f"ERROR: Teacher model not found. Please run teacher training first.")
        return None, None

    student = MnistNetworkStudent().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    
    history = {'steps': [], 'train_accuracies': [], 'test_accuracies': []}

    student.train()
    step = 0
    done = False
    while not done:
        for batch_x, batch_y in train_loader:
            if step >= steps_total: done = True; break
            
            student.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_one_hot = to_one_hot(batch_y)
            
            with torch.no_grad():
                logits_teacher = teacher(batch_x, 1.0, 1.0)
                y_teacher_soft = F.softmax(logits_teacher / temperature, dim=1)
            
            optimizer.zero_grad()
            logits_student = student(batch_x)
            
            y_student_hard = F.softmax(logits_student, dim=1)
            y_student_soft = F.softmax(logits_student / temperature, dim=1)
            
            loss_hard, acc = loss_and_accuracy(y_student_hard, y_one_hot)
            loss_soft, _ = loss_and_accuracy(y_student_soft, y_teacher_soft)
            
            total_loss = loss_hard * alpha + loss_soft * (1 - alpha) * (temperature ** 2)
            total_loss.backward()
            optimizer.step()

            if (step + 1) % verbose_step == 0:
                test_acc_val = calculate_test_accuracy(student, test_loader)

                history['steps'].append(step + 1)
                history['train_accuracies'].append(acc.item())
                history['test_accuracies'].append(test_acc_val)
                
                print(f"Distilled Student | Step {step+1}/{steps_total} | Train Acc: {acc.item():.4f} | Test Acc: {test_acc_val:.4f}")

            step += 1

    os.makedirs(os.path.dirname(STUDENT_SAVE_PATH), exist_ok=True)
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"Distilled student model saved to {STUDENT_SAVE_PATH}")
    return student, history

if __name__ == '__main__':
    teacher_history = train_teacher()

    plot_single_model_history(teacher_history, "Teacher Model")
    
    student_histories = {}
    
    _, baseline_history = train_student_baseline()
    student_histories['Baseline'] = baseline_history
    
    _, distilled_history1 = train_student_distilled(alpha=0.08, temperature=7)
    if distilled_history1:
        student_histories['Distilled (a=0.08, t=7)'] = distilled_history1
    
    _, distilled_history2 = train_student_distilled(alpha=0.3, temperature=3)
    if distilled_history2:
        student_histories['Distilled (a=0.3, t=3)'] = distilled_history2

    plot_student_comparison(student_histories)