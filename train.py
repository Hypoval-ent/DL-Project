import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DepthDataset
from depth_anything.dpt import DepthAnything
import matplotlib.pyplot as plt

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== DATASET =====
DATASET_PATH = "../dataset"
dataset = DepthDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ===== MODEL =====
model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
model = model.to(device)

# ===== FREEZE ENCODER =====
for param in model.pretrained.parameters():
    param.requires_grad = False

# ===== OPTIMIZER + LOSS =====
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.L1Loss()

# ===== TRAINING =====
for epoch in range(10):

    # 👉 Unfreeze after 5 epochs
    # if epoch == 5:
    #     for param in model.pretrained.parameters():
    #         param.requires_grad = True
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    #     print("Encoder unfrozen")

    for i, (img, depth) in enumerate(loader):

        img = img.to(device)
        depth = depth.to(device)

        # ===== FORWARD =====
        pred = model(img)            # [B, H, W]
        pred = pred.unsqueeze(1)     # [B, 1, H, W]

        # ===== LOSS =====
        loss = loss_fn(pred, depth)

        # ===== BACKPROP =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== PRINT PROGRESS =====
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

        # ===== VISUALIZATION (NON-BLOCKING) =====
        if i == 0:
            depth_map = pred[0].squeeze().detach().cpu().numpy()
            plt.imshow(depth_map, cmap='inferno')
            plt.title(f"Epoch {epoch}")
            plt.axis('off')
            plt.pause(0.001)
            plt.close()

    print(f"Epoch {epoch} completed")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), "depth_model.pth")
print("Model saved successfully!")