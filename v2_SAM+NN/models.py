import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel

class LogoCleanerModel(nn.Module):
    """
    Logo Cleaner model using SlimSAM as the backbone and a custom selection head
    for logo segmentation.
    """
    def __init__(self, slim_sam_path="Zigeng/SlimSAM-uniform-50", freeze_backbone=True):
        super(LogoCleanerModel, self).__init__()
        
        # Load SlimSAM model for the image encoder
        self.sam_model = SamModel.from_pretrained(slim_sam_path)
        
        # Freeze the backbone parameters if specified
        if freeze_backbone:
            for param in self.sam_model.vision_encoder.parameters():
                param.requires_grad = False
        
        # The output dimension of the vision encoder (typically 256 for SlimSAM)
        self.backbone_dim = 256
        
        # Define a selection head that takes image embeddings and predicts logo masks
        self.selection_head = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Upsampling module to match the original image dimensions
        # SAM's vision encoder typically downsamples by a factor of 16
        self.upsampler = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
    
    def forward(self, pixel_values):
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Tensor of shape [batch_size, 3, height, width]
                         Preprocessed input images
        
        Returns:
            Tensor of shape [batch_size, 1, height, width]
            Predicted logo masks
        """
        # Extract image embeddings using SlimSAM's vision encoder
        with torch.set_grad_enabled(not self.sam_model.vision_encoder.training):
            image_embeddings = self.sam_model.vision_encoder(pixel_values).last_hidden_state
        
        # Pass image embeddings through the selection head
        mask_logits = self.selection_head(image_embeddings)
        
        # Upsample to match original image dimensions
        logo_masks = self.upsampler(mask_logits)
        
        return logo_masks