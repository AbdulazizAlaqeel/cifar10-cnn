import torch
from torch import nn
from torch.utils.data import DataLoader, Subset 
from torchvision import datasets, transforms
import time

torch.manual_seed(0)

# Settings
AUGMENT = True
BATCH_SIZE = 128
EPOCHS = 40

# Part 1: load the data
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if AUGMENT:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
else:
    train_transform = test_transform

# split the 50,000 training images into 45,000 train + 5,000 validation
full_train_aug   = datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
full_train_plain = datasets.CIFAR10("data", train=True, transform=test_transform)
order = torch.randperm(50000, generator=torch.Generator().manual_seed(0))
val_ids   = order[:5000]
train_ids = order[5000:]
train_data = Subset(full_train_aug, train_ids)
val_data   = Subset(full_train_plain, val_ids)
test_data = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1000)
test_loader = DataLoader(test_data, batch_size=1000)

print("Training pictures:", len(train_data))
print("Validation pictures:", len(val_data))
print("Test pictures:", len(test_data))
print("Classes:", full_train_aug.classes)  # Subset has no .classes
pictures, labels = next(iter(train_loader))
print("One batch:", pictures.shape)


# Part 2: define the model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),    nn.BatchNorm2d(32),  nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

    nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10),
)

print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))
scores = model(pictures)
print("Scores for one batch:", scores.shape)


# Part 3: train and test
def evaluate(loader):  # now takes which images to check
    model.eval()
    correct = 0
    with torch.no_grad():
        for pictures, labels in loader:
            guesses = model(pictures).argmax(dim=1)
            correct += (guesses == labels).sum().item()
    model.train()
    return 100 * correct / len(loader.dataset)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = 0     # best validation accuracy so far
best_epoch = 0

for epoch in range(EPOCHS):
    start = time.time()
    lr_now = optimizer.param_groups[0]["lr"]
    total_loss = 0
    for pictures, labels in train_loader:
        loss = loss_fn(model(pictures), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(val_loader)  # validation, not test
    if val_acc > best_val:                                  # new best?
        best_val = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "cifar_model.pt")    # save the best weights
    print(f"Epoch {epoch + 1}: lr {lr_now:.6f}, loss {avg_loss:.3f}, val accuracy {val_acc:.2f}%, time {time.time() - start:.0f}s")

# load the best model and run the test ONCE
model.load_state_dict(torch.load("cifar_model.pt"))
test_acc = evaluate(test_loader)
print(f"Best epoch: {best_epoch} (val {best_val:.2f}%)")
print(f"FINAL TEST accuracy: {test_acc:.2f}%")