# Logo Detection using SuperPixel and Random Forest

This folder contains the implementation of a logo detection system using SuperPixel segmentation and Random Forest classification.

## Overview

This implementation represents our initial approach to logo detection and removal, using traditional computer vision techniques:

1. SuperPixel segmentation to divide the image into meaningful regions
2. Random Forest classifier to identify logo regions based on extracted features
3. Basic post-processing for logo removal

## Hardware Requirements

- CPU-only machine is sufficient
- No GPU required for training or inference

## Project Structure

```
v0_SuperPixel+Rf/
│
├── dataloader.py       # Data loading and preprocessing functionality
├── logo_rf_model.pkl   # Pre-trained Random Forest model
├── v0.ipynb            # Main notebook with demonstration and training code
└── datasets/           # Dataset directory (you need to email the author download yourself)
    ├── className2ClassID.txt  # Class mapping file
    ├── scripts/        # Utility scripts for data handling
    ├── test/           # Test dataset
    └── train/          # Training dataset
```

## Environment Setup with Conda

1. Create and activate a new conda environment:
   ```bash
   conda create -n logo_rf python=3.12
   conda activate logo_rf
   ```

2. Install required packages:
   ```bash
   conda install numpy pandas scikit-learn scikit-image matplotlib jupyter
   conda install -c conda-forge opencv
   ```

## Usage

1. Navigate to the project directory:
   ```bash
   cd v0_SuperPixel+Rf
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `v0.ipynb` and run the cells to:
   - Load and preprocess data
   - Train the Random Forest model (or use the pre-trained model)
   - Test logo detection
   - Visualize results

## Dataset Format

The dataset is organized with a filelist structure:
- `train/filelist-logosonly.txt`: Training images with logo annotations
- `test/filelist.txt`: Test images with logo annotations

Each line in the filelist contains an image path and corresponding annotation information.

## Model Details

- The Random Forest model (`logo_rf_model.pkl`) is trained on features extracted from SuperPixel segments
- Features include color, texture, and position information
- The model classifies segments as either logo or non-logo regions

## Results

This approach provides a baseline for logo detection with reasonable accuracy for simple logos. However, it has limitations with complex logos or challenging backgrounds, which motivated the development of our more advanced approaches (v1 and v2).

## References

- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Scikit-image SuperPixel Documentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)