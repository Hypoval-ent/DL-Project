import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os

def disparity_to_depth(disparity):
    disparity = disparity.astype(np.float32)
    disparity[disparity == 0] = 0.1
    depth = (2262.52 * 0.209313) / disparity
    return depth

class DepthDataset(Dataset):
    def __init__(self, root):
        self.img_paths = []
        self.disp_paths = []

        img_root = os.path.join(root, "images")
        
        for city in os.listdir(img_root):
            city_path = os.path.join(img_root, city)

            for file in os.listdir(city_path):
                img_path = os.path.join(city_path, file)

                disp_path = img_path.replace("images", "disparity") \
                                    .replace("_leftImg8bit", "_disparity")

                self.img_paths.append(img_path)
                self.disp_paths.append(disp_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        disp = cv2.imread(self.disp_paths[idx], cv2.IMREAD_UNCHANGED)

        depth = disparity_to_depth(disp)

        # normalize
        depth = depth / (depth.max() + 1e-8)

        # resize
        img = cv2.resize(img, (518, 518))
        depth = cv2.resize(depth, (518, 518))

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0).float()

        return img, depth