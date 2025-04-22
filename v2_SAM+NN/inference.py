import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from models import LogoCleanerModel
from torchvision import transforms
from segment_anything import SamPredictor, sam_model_registry

def load_model(model_path="best_model.pth", device="cuda"):
    """
    Load a trained LogoCleanerModel from a checkpoint file.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Loaded LogoCleanerModel in evaluation mode
    """
    # Initialize model
    model = LogoCleanerModel()
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    return model

def preprocess_image(image_path, device="cuda"):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        device: Device to put the tensor on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (preprocessed_tensor, original_size, original_image)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Define preprocessing transforms 
    # This should match the preprocessing used during training
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    return input_tensor, original_size, image

def predict_logo_mask(model, image_tensor, original_size):
    """
    Predict the logo mask for an input image.
    
    Args:
        model: The LogoCleanerModel
        image_tensor: Preprocessed input image tensor
        original_size: Original size of the image (width, height)
    
    Returns:
        Numpy array of the predicted mask, resized to original dimensions
    """
    with torch.no_grad():
        # Get prediction
        mask_pred = model(image_tensor)
        
        # Convert to numpy array
        mask = mask_pred.squeeze().cpu().numpy()
        
        # Resize to original dimensions
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
        mask = np.array(mask_pil)
        
        return mask
def segment_with_sam_using_bbox(image, bbox, sam_model=None, model_type="vit_b", checkpoint="./sam_vit_b_01ec64.pth"):
    """
    Use Segment Anything Model (SAM) to generate a segmentation mask based on a bounding box.
    
    Args:
        image: PIL Image or numpy array (H, W, 3) in RGB format
        bbox: tuple of (x_min, y_min, x_max, y_max) for the bounding box
        sam_model: Pre-loaded SAM model (if None, will load a new one)
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        checkpoint: Path to SAM checkpoint file
        
    Returns:
        numpy array: Binary segmentation mask
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Convert RGB to BGR for OpenCV compatibility if needed
    if image_np.shape[-1] == 3:  # If it has 3 channels
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Initialize SAM if not provided
    if sam_model is None:
        if checkpoint is None:
            raise ValueError("Either sam_model or checkpoint path must be provided")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        if torch.cuda.is_available():
            sam_model.to("cuda")
    
    # Initialize the predictor
    predictor = SamPredictor(sam_model)
    
    # Set the image
    predictor.set_image(image_np)
    
    # Convert bbox to the format expected by SAM
    x_min, y_min, x_max, y_max = bbox
    box = np.array([x_min, y_min, x_max, y_max])
    box_np = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
    
    # Get segmentation masks from SAM
    masks, scores, logits = predictor.predict(
        box=box_np,
        multimask_output=True  # Return multiple masks
    )
    
    # Choose the mask with the highest score
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    
    return best_mask

def predict_and_refine_with_sam(model, image_tensor, original_image, original_size, sam_model=None, sam_checkpoint=None):
    """
    Predict logo mask, extract bbox, then refine using SAM.
    
    Args:
        model: The LogoCleanerModel
        image_tensor: Preprocessed input image tensor
        original_image: Original PIL Image or numpy array
        original_size: Original size of the image (width, height)
        sam_model: Pre-loaded SAM model (optional)
        sam_checkpoint: Path to SAM checkpoint file (optional)
        
    Returns:
        tuple: (initial_mask, refined_mask, bbox)
    """
    # First get the initial mask and bbox using previous functions
    with torch.no_grad():
        # Get prediction
        mask_pred = model(image_tensor)
        
        # Convert to numpy array
        initial_mask = mask_pred.squeeze().cpu().numpy()
        
        # Resize to original dimensions
        mask_pil = Image.fromarray((initial_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
        initial_mask = np.array(mask_pil)
    
    # Get bounding box from mask
    binary_mask = initial_mask > 127
    y_indices, x_indices = np.where(binary_mask)
    
    # If no foreground pixels were found
    if len(y_indices) == 0 or len(x_indices) == 0:
        return initial_mask, None, None
    
    # Calculate bounding box coordinates
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bbox = (x_min, y_min, x_max, y_max)
    
    # Use SAM to refine the segmentation using the bbox
    refined_mask = segment_with_sam_using_bbox(
        original_image,
        bbox,
    )
    
    return initial_mask, refined_mask, bbox

def visualize_and_save_results(original_image, initial_mask, refined_mask, bbox, output_dir="./output"):
    """
    Visualize and save the original image, initial mask, refined mask, and bounding box.
    
    Args:
        original_image: PIL Image of the original input
        initial_mask: Numpy array of initial predicted mask
        refined_mask: Numpy array of SAM-refined mask
        bbox: Tuple of (x_min, y_min, x_max, y_max)
        output_dir: Directory to save the output images
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert original image to numpy if it's a PIL Image
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image.copy()
    
    # Create a copy of original image to draw bounding box
    bbox_image = original_np.copy()
    
    # Draw bounding box
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(bbox_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Convert initial mask to binary and create a colored overlay
    if initial_mask is not None:
        initial_binary = initial_mask > 127
        initial_overlay = original_np.copy()
        initial_overlay[initial_binary] = (
            initial_overlay[initial_binary] * 0.7 + np.array([255, 0, 0]) * 0.3
        ).astype(np.uint8)
    
    # Convert refined mask to binary and create a colored overlay
    if refined_mask is not None:
        refined_binary = refined_mask
        refined_overlay = original_np.copy()
        refined_overlay[refined_binary] = (
            refined_overlay[refined_binary] * 0.7 + np.array([0, 0, 255]) * 0.3
        ).astype(np.uint8)
    
    # Print the bounding box coordinates
    if bbox is not None:
        print(f"Bounding Box (x_min, y_min, x_max, y_max): {bbox}")
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original image with bounding box
    plt.subplot(2, 2, 1)
    plt.imshow(bbox_image)
    plt.title("Original Image with Bounding Box")
    plt.axis("off")
    
    # Original image
    plt.subplot(2, 2, 2)
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")
    
    # Initial mask overlay
    plt.subplot(2, 2, 3)
    if initial_mask is not None:
        plt.imshow(initial_overlay)
        plt.title("Initial Mask (Red)")
    else:
        plt.imshow(original_np)
        plt.title("No Initial Mask Available")
    plt.axis("off")
    
    # Refined mask overlay
    plt.subplot(2, 2, 4)
    if refined_mask is not None:
        plt.imshow(refined_overlay)
        plt.title("Refined SAM Mask (Blue)")
    else:
        plt.imshow(original_np)
        plt.title("No Refined Mask Available")
    plt.axis("off")
    
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig(os.path.join(output_dir, "combined_visualization.png"), dpi=300)
    
    # Also save individual images
    
    # Save original image with bounding box
    plt.figure(figsize=(8, 8))
    plt.imshow(bbox_image)
    plt.title("Original Image with Bounding Box")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_with_bbox.png"), dpi=300)
    plt.close()
    
    # Save original image
    plt.figure(figsize=(8, 8))
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original.png"), dpi=300)
    plt.close()
    
    # Save initial mask overlay
    plt.figure(figsize=(8, 8))
    if initial_mask is not None:
        plt.imshow(initial_overlay)
        plt.title("Initial Mask (Red)")
        
        # Also save the raw mask as grayscale
        mask_image = Image.fromarray((initial_mask).astype(np.uint8))
        mask_image.save(os.path.join(output_dir, "initial_mask_raw.png"))
    else:
        plt.imshow(original_np)
        plt.title("No Initial Mask Available")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "initial_mask_overlay.png"), dpi=300)
    plt.close()
    
    # Save refined mask overlay
    plt.figure(figsize=(8, 8))
    if refined_mask is not None:
        plt.imshow(refined_overlay)
        plt.title("Refined SAM Mask (Blue)")
        
        # Also save the raw mask as grayscale
        refined_mask_uint8 = (refined_mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(refined_mask_uint8)
        mask_image.save(os.path.join(output_dir, "refined_mask_raw.png"))
    else:
        plt.imshow(original_np)
        plt.title("No Refined Mask Available")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refined_mask_overlay.png"), dpi=300)
    plt.close()
    
    print(f"All images saved to {output_dir}")
    
    # Optionally still show the combined figure if in interactive mode
    plt.figure(figsize=(15, 10))
    
    # Original image with bounding box
    plt.subplot(2, 2, 1)
    plt.imshow(bbox_image)
    plt.title("Original Image with Bounding Box")
    plt.axis("off")
    
    # Original image
    plt.subplot(2, 2, 2)
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")
    
    # Initial mask overlay
    plt.subplot(2, 2, 3)
    if initial_mask is not None:
        plt.imshow(initial_overlay)
        plt.title("Initial Mask (Red)")
    else:
        plt.imshow(original_np)
        plt.title("No Initial Mask Available")
    plt.axis("off")
    
    # Refined mask overlay
    plt.subplot(2, 2, 4)
    if refined_mask is not None:
        plt.imshow(refined_overlay)
        plt.title("Refined SAM Mask (Blue)")
    else:
        plt.imshow(original_np)
        plt.title("No Refined Mask Available")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def remove_logo(image, mask, threshold=0.5):
    """
    Remove the logo from an image using the predicted mask.
    
    Args:
        image: Original image (PIL Image or numpy array)
        mask: Predicted logo mask (numpy array)
        threshold: Threshold for binarizing the mask (0-1)
    
    Returns:
        Numpy array of the image with the logo removed
    """
    # Convert mask to binary using threshold
    binary_mask = (mask > threshold * 255).astype(np.uint8)
    
    # Convert image to numpy if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create inpainting mask (1 for logo, 0 for background)
    inpaint_mask = binary_mask * 255
    
    # Use OpenCV inpainting to remove logo
    image_without_logo = cv2.inpaint(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        inpaint_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )
    
    # Convert back to RGB
    image_without_logo = cv2.cvtColor(image_without_logo, cv2.COLOR_BGR2RGB)
    
    return image_without_logo

def main(image_path, model_path="best_model.pth", threshold=0.5):
    """
    Main function for logo removal.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        threshold: Threshold for binarizing the mask (0-1)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Preprocess image
    image_tensor, original_size, original_image = preprocess_image(image_path, device)
    
    # Predict logo mask
    print("Predicting logo mask...")
    # mask = predict_logo_mask(model, image_tensor, original_size)
    initial_mask, mask, bbox = predict_and_refine_with_sam(
        model,
        image_tensor,
        original_image,
        original_size
    )
    print("Logo mask predicted successfully")
    # print the initial mask, refined mask, and bbox based on the image
    visualize_and_save_results(original_image, initial_mask, mask, bbox)
    # # Remove logo
    # print("Removing logo...")
    # result = remove_logo(original_image, mask, threshold)
    
    # # Plot results
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(original_image)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")
    
    # axes[1].imshow(mask, cmap='gray')
    # axes[1].set_title("Predicted Logo Mask")
    # axes[1].axis("off")
    
    # axes[2].imshow(result)
    # axes[2].set_title("Image with Logo Removed")
    # axes[2].axis("off")
    
    # plt.tight_layout()
    # plt.savefig("logo_removal_result.png")
    # print("Results saved as 'logo_removal_result.png'")
    
    # # Save result
    # Image.fromarray(result).save("cleaned_image.png")
    # print(f"Cleaned image saved as 'cleaned_image.png'")
    
    # return mask, result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove logos from images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (0-1)")
    
    args = parser.parse_args()
    main(args.image, args.model, args.threshold)