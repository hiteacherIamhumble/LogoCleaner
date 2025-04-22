# Logo Detection using U-Net Architecture

This folder contains the implementation of a deep learning-based logo detection system using U-Net architecture for semantic segmentation.

## Overview

This implementation represents our second approach to logo detection, using deep learning for more accurate segmentation:

1. U-Net architecture for pixel-level logo segmentation
2. Custom loss functions for handling imbalanced data (logos are typically small objects)
3. Training pipeline with data augmentation and validation

## Hardware Requirements

- GPU is required for efficient training (our team used an NVIDIA RTX 4090)
- CPU-only inference is possible but significantly slower

## Project Structure

```
v1_Unet/
│
├── dataloader.py       # Custom PyTorch dataset and data loading utilities
├── loss_functions.py   # Implementation of custom loss functions
├── models.py           # U-Net model architecture implementation
├── pipeline.ipynb      # Main notebook with training and inference pipeline
├── trainer.py          # Training loop and validation logic
├── utils.py            # Utility functions for visualization and metrics
├── checkpoints/        # Model checkpoints directory
│   ├── best_model.pth  # Best model weights based on validation
│   └── latest_checkpoint.pth  # Latest training checkpoint
├── results/            # Results directory for storing outputs
│   └── logo_detection_model.pth  # Final trained model
├── test_img/           # Sample test images
│   ├── test1.png
│   └── test2.png
└── wandb/              # Weights & Biases logging directory
```

## Environment Setup with Conda

1. Create and activate a new conda environment:
   ```bash
   conda create -n logo_unet python=3.10
   conda activate logo_unet
   ```

2. Install PyTorch with GPU support (This is verified on Linux Server):
   ```bash
   pip3 install torch torchvision torchaudio
   ```
   Note: This download the CUDA version (12.4).

3. Install additional dependencies:
   ```bash
   conda install matplotlib opencv numpy pillow scikit-learn scikit-image jupyter
   pip install wandb  # For experiment tracking
   ```

## Usage

1. Navigate to the project directory:
   ```bash
   cd v1_Unet
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Download the Unet model from the huggingface
   - Model link: [LogoCleaner Model on HuggingFace](https://huggingface.co/PeterDAI/LogoCleaner/tree/main)
  - Place the downloaded model in the result directory.

4. Open `pipeline.ipynb` and run the cells to:
   - Configure dataset paths and hyperparameters
   - Load and preprocess data
   - Train the U-Net model
   - Evaluate results on the validation set
   - Run inference on test images

## Training Details

- The training pipeline uses a U-Net architecture optimized for logo segmentation
- Training parameters are configurable in the notebook:
  - Image size: 512px
  - Batch size: 4
  - Learning rate: 0.0001
  - Number of epochs: 30
- Progress is logged using Weights & Biases for experiment tracking

## Dataset Format

The dataset follows the same format as in v0:
- `train/filelist-logosonly.txt`: Training images with logo annotations
- `test/filelist.txt`: Test images with logo annotations

## Results

This approach provides significantly improved logo detection compared to the SuperPixel+RF approach, with better handling of complex logos and challenging backgrounds. The U-Net architecture enables pixel-precise segmentation masks, which are essential for high-quality logo removal.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)