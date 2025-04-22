# Logo Detection using SAM and Neural Networks

This folder contains the implementation of our most advanced approach to logo detection and removal, combining Facebook's Segment Anything Model (SAM) with a custom neural network for logo detection.

## Overview

This implementation represents our final, most sophisticated approach to logo detection and removal:

1. SAM (Segment Anything Model) for high-quality image segmentation
2. Custom neural network for logo/non-logo classification of segments
3. Advanced post-processing for seamless logo removal
4. Integration pipeline for end-to-end logo detection and removal

## Hardware Requirements

- GPU is required for efficient training and inference (our team used an NVIDIA RTX 4090)
- At least 16GB of VRAM recommended for running the SAM model efficiently

## Project Structure

```
v2_SAM+NN/
│
├── best_model.pth       # need to be installed from huggingface
├── dataloader.py        # Data loading and preprocessing
├── inference.py         # Inference pipeline for logo detection
├── main.py              # Main script for training and evaluation
├── models.py            # Model architecture definitions
├── requirements.txt     # Python package dependencies
├── sam_vit_b_01ec64.pth # SAM model checkpoint (need to be installed from huggingface)
├── train.py             # Training script
└── checkpoints/         # Directory for saving model checkpoints
```

## Environment Setup with Conda

1. Create and activate a new conda environment:
   ```bash
   conda create -n logo_sam python=3.10
   conda activate logo_sam
   ```

2. Install PyTorch with GPU support (This is verified on Linux Server):
   ```bash
   pip3 install torch torchvision torchaudio
   ```
   Note: This download the CUDA version (12.4).

3. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the model checkpoint and SAM model checkpoint:
   - Model link: [LogoCleaner Model on HuggingFace](https://huggingface.co/PeterDAI/LogoCleaner/tree/main)
   - Place the downloaded model in the current directory.

## Usage

### Training

1. Navigate to the project directory:
   ```bash
   cd v2_SAM+NN
   ```

2. Run the training script and you can adjust the epoch and learning rate:
   ```bash
   python main.py train --epochs 50 --lr 1e-4
   ```

### Inference

1. For logo detection and removal on new images:
   ```bash
   python main.py infer --image test2.png --model best_model.pth
   ```


## SAM Integration

This approach leverages Facebook's Segment Anything Model (SAM) to generate high-quality segments, which are then classified by our custom neural network. This combination provides several advantages:

1. SAM handles the complex task of segmentation with state-of-the-art performance
2. Our custom neural network focuses specifically on logo classification
3. The pipeline includes post-processing for natural-looking logo removal

## Results

This approach achieves the best results among all our implementations, with high accuracy in logo detection and natural-looking removal. The visualizations in the `output/` directory demonstrate the quality of the results.

## References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)