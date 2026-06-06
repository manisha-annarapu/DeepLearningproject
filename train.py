import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim

print("Deep Learning Project Started")

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./dataset",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Completed")

torch.save(model.state_dict(), "model/mnist_model.pth")

print("Training Completed Successfully")
print("Model Saved in model folder")