import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tkinter as tk
from PIL import Image, ImageTk
import random

class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.layer_config = 0
        self.build_model()

    def build_model(self):
        if self.layer_config == 0:
            self.layers = nn.ModuleList([
                nn.Linear(28 * 28, 10)
            ])
        elif self.layer_config == 1:
            self.layers = nn.ModuleList([
                nn.Linear(28 * 28, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            ])
        elif self.layer_config == 2:
            self.layers = nn.ModuleList([
                nn.Linear(28 * 28, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ])
        elif self.layer_config == 3:
            self.layers = nn.ModuleList([
                nn.Linear(28 * 28, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            ])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.layers:
            x = layer(x)
        return x 

    def add_layer(self):
        if self.layer_config < 3:
            self.layer_config += 1
            self.build_model()
            self.to(device)  # move new layers to correct device

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.2)
criterion = nn.CrossEntropyLoss()

# GUI Setup
window = tk.Tk()
window.title("MNIST Classifier Progress")

canvas = tk.Canvas(window, width=280, height=280)
canvas.pack()

info_label = tk.Label(window, text="Training Progress")
info_label.pack()

progress_label = tk.Label(window, text="Epoch: 0/0 | Batch: 0/0 | Accuracy: 0%")
progress_label.pack()

accuracy_label = tk.Label(window, text="Final Accuracy: N/A")
accuracy_label.pack()

progress_bar = tk.Canvas(window, width=300, height=20, bg="white")
progress_bar.pack()

def update_progress_bar(progress):
    progress_bar.delete("all")
    progress_bar.create_rectangle(0, 0, progress * 3, 20, fill="green")

def update_display(image, prediction, true_label):
    img = Image.fromarray((image.squeeze().numpy() * 255).astype('uint8'))
    img = ImageTk.PhotoImage(img.resize((280, 280)))
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img
    info_label.config(text=f"Prediction: {prediction}, Actual: {true_label}")
    window.update()

def add_noise(data, noise_factor):
    noise = torch.randn_like(data) * noise_factor
    noisy_data = data + noise
    return torch.clamp(noisy_data, 0., 1.)

def train_model():
    global optimizer, scheduler

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        noise_factor = max(0, 0.5 - epoch * 0.05)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = add_noise(data, noise_factor)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                accuracy = 100 * correct / total
                progress_label.config(text=f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Accuracy: {accuracy:.2f}%")
                update_progress_bar(batch_idx / len(train_loader) * 100)

                sample_idx = random.randint(0, len(data) - 1)
                update_display(data[sample_idx].cpu(), predicted[sample_idx].item(), target[sample_idx].item())

        scheduler.step()

        if epoch in [2, 5, 8]:  # Add layers at specific epochs
            model.add_layer()
            optimizer = optim.SGD(model.parameters(), lr=scheduler.get_last_lr()[0])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.2)

    test_model()

def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    accuracy_label.config(text=f"Final Accuracy: {accuracy:.2f}%")
    display_random_samples()

def display_random_samples():
    model.eval()
    samples_window = tk.Toplevel(window)
    samples_window.title("Random Test Samples")

    for i in range(10):
        idx = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]

        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1, keepdim=True).item()

        img = Image.fromarray((image.squeeze().numpy() * 255).astype('uint8'))
        img = ImageTk.PhotoImage(img.resize((100, 100)))

        frame = tk.Frame(samples_window)
        frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)

        img_label = tk.Label(frame, image=img)
        img_label.image = img
        img_label.pack()

        text_label = tk.Label(frame, text=f"Pred: {prediction}, True: {label}")
        text_label.pack()

train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.pack()

window.mainloop()
