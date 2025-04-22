import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class LogoDetectionTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        num_epochs=30,
        device=None,
        checkpoint_dir='./checkpoints',
        use_wandb=True,
        wandb_project='logo-detection',
        wandb_entity=None,
        wandb_name=None,
        wandb_config=None
    ):
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model and data
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Loss function
        self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
        
        # Optimizer and scheduler
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=1e-4)
        self.lr_scheduler = lr_scheduler
        
        # Training settings
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Weights & Biases integration
        self.use_wandb = use_wandb
        if use_wandb:
            self.init_wandb(wandb_project, wandb_entity, wandb_name, wandb_config)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
    def init_wandb(self, project, entity, name, config):
        """
        Initialize Weights & Biases for experiment tracking
        """
        if not name:
            name = f"unet-logo-{time.strftime('%Y%m%d-%H%M%S')}"
            
        if not config:
            config = {
                'model_type': 'UNet',
                'epochs': self.num_epochs,
                'batch_size': self.train_dataloader.batch_size,
                'optimizer': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'criterion': self.criterion.__class__.__name__,
            }
            
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config
        )
        
        # Watch model
        wandb.watch(self.model, log='all')
    
    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        # Create progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Skip empty batches
                if 'image' not in batch or 'mask' not in batch:
                    print(f"Warning: Missing data in batch {batch_idx}, skipping")
                    continue
                
                # Check if batch is empty
                if batch['image'].numel() == 0 or batch['mask'].numel() == 0:
                    print(f"Warning: Empty tensors in batch {batch_idx}, skipping")
                    continue
                
                # Get data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Verify shapes
                if len(images.shape) != 4 or len(masks.shape) != 4:
                    print(f"Warning: Incorrect tensor dimensions in batch {batch_idx}, skipping")
                    print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss encountered in batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate Dice coefficient
                with torch.no_grad():
                    preds = torch.sigmoid(outputs) > 0.5
                    dice = self.calculate_dice(preds, masks > 0.5)
                    
                # Update metrics
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice.item():.4f}"
                })
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Batch structure: {list(batch.keys())}")
                continue
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_dice': dice.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Count processed batches for accurate averaging
        processed_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Skip empty batches
                if 'image' not in batch or 'mask' not in batch:
                    print(f"Warning: Missing data in batch {batch_idx}, skipping")
                    continue
                
                # Check if batch is empty
                if batch['image'].numel() == 0 or batch['mask'].numel() == 0:
                    print(f"Warning: Empty tensors in batch {batch_idx}, skipping")
                    continue
                
                # Get data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Verify shapes
                if len(images.shape) != 4 or len(masks.shape) != 4:
                    print(f"Warning: Incorrect tensor dimensions in batch {batch_idx}, skipping")
                    print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss encountered in batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate Dice coefficient
                with torch.no_grad():
                    preds = torch.sigmoid(outputs) > 0.5
                    dice = self.calculate_dice(preds, masks > 0.5)
                    
                # Update metrics
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                processed_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice.item():.4f}"
                })
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Batch structure: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
                continue
                
        # Calculate epoch metrics
        if processed_batches > 0:
            epoch_loss /= processed_batches
            epoch_dice /= processed_batches
        else:
            print("Warning: No valid batches were processed during this epoch")
            epoch_loss = float('inf')
            epoch_dice = 0.0
        
        return epoch_loss, epoch_dice
    
    def validate(self, epoch):
        """
        Validate the model on the validation set
        """
        if not self.val_dataloader:
            return None, None
            
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        # Create progress bar
        pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Valid]")
        
        # Sample images for visualization
        vis_images = []
        vis_masks = []
        vis_preds = []
        max_samples = 4  # Number of samples to visualize
        
        processed_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Skip empty batches
                    if 'image' not in batch or 'mask' not in batch:
                        print(f"Warning: Missing data in validation batch {batch_idx}, skipping")
                        continue
                    
                    # Check if batch is empty
                    if batch['image'].numel() == 0 or batch['mask'].numel() == 0:
                        print(f"Warning: Empty tensors in validation batch {batch_idx}, skipping")
                        continue
                    
                    # Get data
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    # Verify shapes
                    if len(images.shape) != 4 or len(masks.shape) != 4:
                        print(f"Warning: Incorrect tensor dimensions in validation batch {batch_idx}, skipping")
                        print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
                        continue
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, masks)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss encountered in validation batch {batch_idx}, skipping")
                        continue
                    
                    # Calculate Dice coefficient
                    preds = torch.sigmoid(outputs) > 0.5
                    dice = self.calculate_dice(preds, masks > 0.5)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_dice += dice.item()
                    processed_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'val_loss': f"{loss.item():.4f}",
                        'val_dice': f"{dice.item():.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error processing validation batch {batch_idx}: {e}")
                    continue
                
                # Collect samples for visualization
                if batch_idx == 0 and self.use_wandb:
                    for i in range(min(images.size(0), max_samples)):
                        vis_images.append(images[i].cpu())
                        vis_masks.append(masks[i].cpu())
                        vis_preds.append(torch.sigmoid(outputs[i]).cpu())
            
            # Calculate validation metrics
            if processed_batches > 0:
                val_loss /= processed_batches
                val_dice /= processed_batches
            else:
                print("Warning: No valid batches were processed during validation")
                val_loss = float('inf')
                val_dice = 0.0
            
            # Visualize predictions if using wandb
            if self.use_wandb and vis_images:
                self.log_predictions(vis_images, vis_masks, vis_preds, epoch)
                
        return val_loss, val_dice
    
    def calculate_dice(self, preds, targets, smooth=1e-6):
        """
        Calculate Dice coefficient
        Formula: 2*|X∩Y|/(|X|+|Y|)
        """
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        # Calculate Dice
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return dice
    
    def log_predictions(self, images, masks, preds, epoch):
        """
        Log prediction visualizations to wandb
        """
        num_samples = len(images)
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(num_samples):
            # Denormalize image
            img = images[i].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Get mask and prediction
            mask = masks[i].permute(1, 2, 0).numpy()
            pred = preds[i].permute(1, 2, 0).numpy()
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask[:, :, 0], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred[:, :, 0], cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        wandb.log({f"predictions_epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice
        }
        
        # Save the latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save checkpoint for this epoch
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save the best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"Saved best model with Dice score: {self.best_val_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_dice = checkpoint['best_val_dice']
        
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
    
    def train(self, resume_from=None):
        """
        Train the model for the specified number of epochs
        """
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"Starting training from epoch {start_epoch + 1}")
        
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            # Train for one epoch
            train_loss, train_dice = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_dice = self.validate(epoch)
            
            # Update learning rate
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            if val_loss is not None:
                metrics['val_loss'] = val_loss
                metrics['val_dice'] = val_dice
                
                # Check if this is the best model
                is_best = False
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.best_val_loss = val_loss
                    is_best = True
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
            
            # Log to console
            log_msg = f"Epoch {epoch+1}/{self.num_epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}"
            
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
                
            print(log_msg)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(metrics)
        
        # Finish
        print("Training complete!")
        if self.use_wandb:
            wandb.finish()
    
    def predict(self, image_tensor):
        """
        Make a prediction for a single image
        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
            output = self.model(image_tensor)
            pred = torch.sigmoid(output)
            
        return pred