import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from models import LogoCleanerModel

def train_logo_cleaner(dataloader, val_dataloader=None, 
                       num_epochs=50, learning_rate=1e-4, 
                       weight_decay=1e-5, device="cuda"):
    """
    Train the logo cleaner model.
    
    Args:
        dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to use for training ('cuda' or 'cpu')
    
    Returns:
        Trained model
    """
    # Initialize wandb
    wandb.init(project="LogoCleaner-Apr17", name="slimSAM-logo-segmentation")
    config = wandb.config
    config.learning_rate = learning_rate
    config.weight_decay = weight_decay
    config.epochs = num_epochs
    
    # Get batch size from dataloader
    sample_batch = next(iter(dataloader))
    config.batch_size = sample_batch['image'].shape[0]
    
    # Initialize model
    model = LogoCleanerModel(freeze_backbone=True)
    model = model.to(device)
    wandb.watch(model, log="all")
    
    # Define loss functions
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.smooth = smooth
            
        def forward(self, predictions, targets):
            # Flatten predictions and targets
            predictions = predictions.view(-1)
            targets = targets.view(-1)
            
            # Calculate intersection and union
            intersection = (predictions * targets).sum()
            union = predictions.sum() + targets.sum()
            
            # Calculate Dice coefficient and loss
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice
    
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    
    # Define optimizer - only for trainable parameters
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice_loss = 0.0
        train_bce_loss = 0.0
        
        # Create progress bar
        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_pbar:
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Ensure outputs and masks have the same size
            if outputs.shape != masks.shape:
                outputs = F.interpolate(
                    outputs, 
                    size=masks.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate loss (combined BCE and Dice loss)
            bce = bce_criterion(outputs, masks)
            dice = dice_criterion(outputs, masks)
            loss = bce + dice
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_bce_loss += bce.item()
            train_dice_loss += dice.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": loss.item(),
                "bce": bce.item(),
                "dice": dice.item()
            })
        
        # Calculate average losses for the epoch
        avg_train_loss = train_loss / len(dataloader)
        avg_train_bce = train_bce_loss / len(dataloader)
        avg_train_dice = train_dice_loss / len(dataloader)
        
        # Validation phase (if validation data is provided)
        val_loss = 0.0
        val_bce_loss = 0.0
        val_dice_loss = 0.0
        val_iou = 0.0
        
        if val_dataloader is not None:
            model.eval()
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_pbar:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    outputs = model(images)
                    
                    # Ensure outputs and masks have the same size
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(
                            outputs, 
                            size=masks.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    # Calculate losses
                    bce = bce_criterion(outputs, masks)
                    dice = dice_criterion(outputs, masks)
                    loss = bce + dice
                    
                    # Calculate IoU (Intersection over Union)
                    pred_masks = (outputs > 0.5).float()
                    intersection = (pred_masks * masks).sum((1, 2, 3))
                    union = pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
                    batch_iou = (intersection / (union + 1e-7)).mean().item()
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_bce_loss += bce.item()
                    val_dice_loss += dice.item()
                    val_iou += batch_iou
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        "loss": loss.item(),
                        "iou": batch_iou
                    })
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_bce = val_bce_loss / len(val_dataloader)
            avg_val_dice = val_dice_loss / len(val_dataloader)
            avg_val_iou = val_iou / len(val_dataloader)
            
            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/bce_loss": avg_train_bce,
                "train/dice_loss": avg_train_dice,
                "val/loss": avg_val_loss,
                "val/bce_loss": avg_val_bce,
                "val/dice_loss": avg_val_dice,
                "val/iou": avg_val_iou,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Save best model based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'iou': avg_val_iou,
                }, "best_model.pth")
                print(f"New best model saved with validation loss: {best_loss:.6f}, IoU: {avg_val_iou:.6f}")
        else:
            # Log metrics (train only)
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/bce_loss": avg_train_bce,
                "train/dice_loss": avg_train_dice,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })
            
            # Update learning rate scheduler
            scheduler.step(avg_train_loss)
            
            # Save best model based on training loss
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, "best_model.pth")
                print(f"New best model saved with training loss: {best_loss:.6f}")
    
    # Finish wandb run
    wandb.finish()
    
    return model


# Example usage if this script is run directly
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # As mentioned by the user, they already have dataloader set up
    # This is where you would use your dataloader:
    # train_logo_cleaner(dataloader, val_dataloader, device=device)
    
    print("This script contains training functions. Import and call train_logo_cleaner() with your dataloader.")