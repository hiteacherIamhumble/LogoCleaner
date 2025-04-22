import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Formula: 1 - 2*|X∩Y|/(|X|+|Y|)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten the predictions and targets
        batch_size = probs.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice Loss
        return 1 - dice.mean()

class IoULoss(nn.Module):
    """
    IoU Loss for segmentation tasks.
    Formula: 1 - |X∩Y|/|X∪Y|
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten the predictions and targets
        batch_size = probs.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1) - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU Loss
        return 1 - iou.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, logits, targets):
        # Get binary cross entropy
        bce_loss = self.bce(logits, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal weight
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply weight
        focal_loss = self.alpha * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function: weighted sum of BCE, Dice and IoU losses.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, iou_weight=0.0, focal_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.focal = FocalLoss()
        
    def forward(self, logits, targets):
        loss = 0
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
        
        if self.iou_weight > 0:
            loss += self.iou_weight * self.iou(logits, targets)
            
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(logits, targets)
        
        return loss