import torch
import cv2
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== LOAD IMAGE =====
img_path = "frankfurt_000000_000294_leftImg8bit.png"   # 👉 change this
img = cv2.imread(img_path)
img = cv2.resize(img, (518, 518))

img_tensor = torch.tensor(img).permute(2,0,1).float()/255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

# ===== PRETRAINED MODEL =====
model_pre = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
model_pre = model_pre.to(device)
model_pre.eval()

# ===== FINETUNED MODEL =====
model_ft = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
model_ft.load_state_dict(torch.load("depth_model.pth", map_location=device))
model_ft = model_ft.to(device)
model_ft.eval()

# ===== INFERENCE =====
with torch.no_grad():
    pred_pre = model_pre(img_tensor)
    pred_ft = model_ft(img_tensor)

depth_pre = pred_pre[0].cpu().numpy()
depth_ft = pred_ft[0].cpu().numpy()

# ===== NORMALIZE FOR DISPLAY =====
depth_pre = (depth_pre - depth_pre.min()) / (depth_pre.max() - depth_pre.min() + 1e-8)
depth_ft = (depth_ft - depth_ft.min()) / (depth_ft.max() - depth_ft.min() + 1e-8)

# ===== VISUALIZATION =====
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(depth_pre, cmap='inferno')
plt.title("Pretrained")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(depth_ft, cmap='inferno')
plt.title("Fine-tuned")
plt.axis('off')

plt.tight_layout()
plt.show()

# ===== SAVE OUTPUTS =====
cv2.imwrite("depth_pretrained.png", depth_pre * 255)
cv2.imwrite("depth_finetuned.png", depth_ft * 255)

print("Saved outputs: depth_pretrained.png, depth_finetuned.png")