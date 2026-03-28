# Depth Anything Fine-Tuning on Cityscapes

## Overview
This project fine-tunes the Depth Anything model for urban scene depth estimation using the Cityscapes dataset.

## Objective
- Improve depth prediction for urban environments
- Adapt pretrained model to domain-specific data

## Project Structure
project/
 ├── Depth-Anything/
 │    ├── train.py
 │    ├── dataset.py
 │    ├── infer.py
 ├── dataset/
 │    ├── images/
 │    ├── disparity/
 ├── depth_model.pth
 ├── test.jpg

## Methodology
- Convert disparity to depth
- Use pretrained Depth Anything model
- Train decoder while freezing encoder

## Training
Run:
python train.py

## Inference
Run:
python infer.py

## Results
- Improved depth prediction for roads, vehicles, and buildings
- Better structural understanding

## Conclusion
Fine-tuning improves domain-specific depth estimation significantly.

## Author
Debadatta Sahoo
