import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from vit import load_pos_embed_mat, ViT
from matplotlib import pyplot as plt

load_pos_embed_mat()


# Loading data
dataset = datasets.MNIST(root='./../datasets', train=False, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(dataset, shuffle=True, batch_size=256)

# Defining model and training options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(image_size=28, patch_size=7, num_classes=10, linear_map_dim=24,head_n=8, mlp_dim=64)
#model = model.to(device)
num_epochs = 2
lr = 0.005

# Training loop
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()
losses = []
for epoch in range(num_epochs):
    train_loss = 0.0
    for it, batch in enumerate(data_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x).to(device)
        loss = criterion(y_hat, y)

        loss_val = loss.detach().cpu().item()
        train_loss += loss_val / len(data_loader)
        losses.append(loss_val)

        print(it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss:.2f}")

plt.plot(losses)
plt.show()