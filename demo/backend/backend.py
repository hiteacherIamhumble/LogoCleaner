import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import io
import requests
import traceback
from models import LogoCleanerModel
from torchvision import transforms
from segment_anything import SamPredictor, sam_model_registry
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def remove_logo(image, mask, threshold=0.5):
    """
    Remove the logo from an image using the predicted mask.
    
    Args:
        image: Original image (PIL Image)
        mask: Predicted logo mask (numpy array or PIL Image)
        threshold: Threshold for binarizing the mask (0-1)
    
    Returns:
        PIL Image of the image with the logo removed
    """
    
    # === Step 0: Setup ===
    client = openai.OpenAI(api_key="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",
                          max_retries=3,  # Add retries for robustness
                          timeout=300.0)  # 5 minute timeout
    openai_image_size = (512, 512)  # or (1024, 1024) if needed

    # === Step 1: Load original image and mask ===
    original_image = image.convert("RGBA") if isinstance(image, Image.Image) else Image.fromarray(image)
    original_mask = mask.convert("L") if isinstance(mask, Image.Image) else Image.fromarray((mask * 255).astype(np.uint8))
    original_size = original_image.size  # Save for later

    # === Step 2: Resize for OpenAI API ===
    resized_image = original_image.resize(openai_image_size, Image.LANCZOS)
    resized_mask = original_mask.resize(openai_image_size, Image.NEAREST)

    # === Step 3: Apply mask to image (make masked area transparent) ===
    np_img = np.array(resized_image)
    np_msk = np.array(resized_mask)
    np_img[np_msk == 255] = [0, 0, 0, 0]  # transparent

    # Create temporary files for the API
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    masked_image_path = os.path.join(temp_dir, "masked_image.png")
    image_path = os.path.join(temp_dir, "image.png")
    
    masked_image = Image.fromarray(np_img)
    masked_image.save(masked_image_path)
    resized_image.save(image_path)

    # === Step 4: Send to OpenAI for inpainting ===
    logger.info("Sending to OpenAI for inpainting")
    
    try:
        # Attempt with longer timeout
        response = client.images.edit(
            image=open(image_path, "rb"),
            mask=open(masked_image_path, "rb"),
            prompt="fill the blank with background",
            n=1,
            size=f"{openai_image_size[0]}x{openai_image_size[1]}",
            response_format="url"
        )

        # === Step 5: Load and resize back the result ===
        inpainted_url = response.data[0].url
        inpainted_image = Image.open(io.BytesIO(requests.get(inpainted_url).content)).convert("RGBA")
        restored_image = inpainted_image.resize(original_size, Image.LANCZOS)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fallback to a simple blurring approach if OpenAI fails
        logger.info("Falling back to blur-based logo removal")
        restored_image = simple_blur_removal(original_image, original_mask)

    # Clean up temporary files
    if os.path.exists(masked_image_path):
        os.remove(masked_image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
        
    return restored_image

def simple_blur_removal(image, mask, blur_radius=20):
    """
    A fallback method that uses blurring to remove logos when the OpenAI API fails.
    
    Args:
        image: Original image (PIL Image)
        mask: Binary mask of the logo area (PIL Image)
        blur_radius: Intensity of the blur effect
    
    Returns:
        PIL Image with the logo blurred out
    """
    # Convert mask to binary if not already
    if isinstance(mask, np.ndarray):
        binary_mask = mask > 127
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    else:
        mask_img = mask
        binary_mask = np.array(mask) > 127
    
    # Create a blurred version of the entire image
    blurred_img = image.copy().filter(Image.GaussianBlur(blur_radius))
    
    # Convert both to numpy arrays for manipulation
    img_array = np.array(image)
    blur_array = np.array(blurred_img)
    
    # Get the bounding box of the mask to work within that region
    if np.any(binary_mask):
        y_indices, x_indices = np.where(binary_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Replace the masked region with the blurred version
        # Create a mask array properly shaped for broadcasting
        mask_3d = np.expand_dims(binary_mask, axis=2)
        if img_array.shape[2] == 4:  # RGBA
            mask_3d = np.repeat(mask_3d, 4, axis=2)
        else:  # RGB
            mask_3d = np.repeat(mask_3d, 3, axis=2)
            
        # Apply the mask: use blurred image where mask is True
        result_array = np.where(mask_3d, blur_array, img_array)
        
        # Create the final image
        result_img = Image.fromarray(result_array.astype(np.uint8))
        return result_img
    else:
        # If no mask was found, return the original image
        return image
    
def visualize_and_save_results(original_image, initial_mask, refined_mask, bbox, output_dir="./output"):
    """
    Save the original image, initial mask, refined mask, and bounding box without displaying.
    
    Args:
        original_image: PIL Image of the original input
        initial_mask: Numpy array of initial predicted mask
        refined_mask: Numpy array of SAM-refined mask
        bbox: Tuple of (x_min, y_min, x_max, y_max)
        output_dir: Directory to save the output images
    """
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
    
    # Log bbox coordinates instead of printing
    if bbox is not None:
        logger.info(f"Bounding Box (x_min, y_min, x_max, y_max): {bbox}")
    
    # Save individual images directly without using matplotlib
    
    # Save original image with bounding box
    Image.fromarray(bbox_image).save(os.path.join(output_dir, "original_with_bbox.png"))
    
    # Save original image
    Image.fromarray(original_np).save(os.path.join(output_dir, "original.png"))
    
    # Save initial mask overlay
    if initial_mask is not None:
        Image.fromarray(initial_overlay).save(os.path.join(output_dir, "initial_mask_overlay.png"))
        # Also save the raw mask as grayscale
        mask_image = Image.fromarray((initial_mask).astype(np.uint8))
        mask_image.save(os.path.join(output_dir, "initial_mask_raw.png"))
    else:
        Image.fromarray(original_np).save(os.path.join(output_dir, "initial_mask_overlay.png"))
    
    # Save refined mask overlay
    if refined_mask is not None:
        Image.fromarray(refined_overlay).save(os.path.join(output_dir, "refined_mask_overlay.png"))
        # Also save the raw mask as grayscale
        refined_mask_uint8 = (refined_mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(refined_mask_uint8)
        mask_image.save(os.path.join(output_dir, "refined_mask_raw.png"))
    else:
        Image.fromarray(original_np).save(os.path.join(output_dir, "refined_mask_overlay.png"))
    
    logger.info(f"All images saved to {output_dir}")

def logo_detection(model, image_path, output_dir, device="cuda"):
    """
    Step 1: Preprocess, detect logo and generate masks.
    
    Args:
        model: The loaded ML model
        image_path: Path to the input image
        output_dir: Directory to save the output images
        device: Device to use for processing
    """
    try:
        # 1. Preprocess
        logger.info(f"Preprocessing image: {image_path}")
        image_tensor, original_size, original_image = preprocess_image(image_path, device)
        
        # 2. Detect logo and get masks
        logger.info("Detecting logo and creating masks")
        initial_mask, refined_mask, bbox = predict_and_refine_with_sam(
            model, image_tensor, original_image, original_size
        )
        
        # Save original image for reference
        original_image.save(os.path.join(output_dir, "original.png"))
        
        # 3. Generate visualization images
        logger.info("Generating visualization images")
        visualize_and_save_results(original_image, initial_mask, refined_mask, bbox, output_dir)
        
        return True, refined_mask
    except Exception as e:
        logger.error(f"Error in logo detection: {e}")
        logger.error(traceback.format_exc())
        return False, None
        
def generate_image(model, image_path, output_dir, device="cuda", threshold=0.5):
    """
    Step 2: Generate the final logo-removed image using remove_logo.
    
    Args:
        model: The loaded ML model
        image_path: Path to the input image
        output_dir: Directory to save the output images
        device: Device to use for processing
        threshold: Threshold for binarizing the mask (0-1)
    """
    try:
        # Load the original image
        original_image = Image.open(os.path.join(output_dir, "original.png")).convert("RGB")
        
        # Try to load the refined mask, if it exists
        refined_mask_path = os.path.join(output_dir, "refined_mask_raw.png")
        initial_mask_path = os.path.join(output_dir, "initial_mask_raw.png")
        
        if os.path.exists(refined_mask_path):
            mask_image = Image.open(refined_mask_path).convert("L")
        elif os.path.exists(initial_mask_path):
            mask_image = Image.open(initial_mask_path).convert("L")
        else:
            # If no mask exists (rare case), detect it again
            logger.info("No mask found, detecting logo again")
            success, refined_mask = logo_detection(model, image_path, output_dir, device)
            if not success or refined_mask is None:
                logger.error("Failed to detect logo")
                return False
            # Convert mask to PIL image if it's a numpy array
            if isinstance(refined_mask, np.ndarray):
                if refined_mask.dtype == bool:
                    mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
                else:
                    # If mask is in range 0-1, convert to 0-255
                    if refined_mask.max() <= 1.0:
                        mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
                    else:
                        mask_image = Image.fromarray(refined_mask.astype(np.uint8))
            else:
                mask_image = refined_mask
        
        # 4. Remove logo
        logger.info("Removing logo")
        result = remove_logo(original_image, mask_image, threshold)
        result.save(os.path.join(output_dir, "logo_removed.png"))
        logger.info("Logo removal completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error in generating result: {e}")
        logger.error(traceback.format_exc())
        return False

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
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    logger.info("Model loaded successfully")
    
    # Preprocess image
    image_tensor, original_size, original_image = preprocess_image(image_path, device)
    
    # Predict logo mask
    logger.info("Predicting logo mask...")
    initial_mask, refined_mask, bbox = predict_and_refine_with_sam(
        model,
        image_tensor,
        original_image,
        original_size
    )
    logger.info("Logo mask predicted successfully")
    
    # Create output directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results without displaying
    visualize_and_save_results(original_image, initial_mask, refined_mask, bbox, output_dir)
    
    # Use the refined mask if available, otherwise use initial mask
    final_mask = refined_mask if refined_mask is not None else initial_mask
    
    # Convert mask to PIL image if it's a numpy array
    if isinstance(final_mask, np.ndarray):
        if final_mask.dtype == bool:
            mask_image = Image.fromarray((final_mask * 255).astype(np.uint8))
        else:
            # If mask is in range 0-1, convert to 0-255
            if final_mask.max() <= 1.0:
                mask_image = Image.fromarray((final_mask * 255).astype(np.uint8))
            else:
                mask_image = Image.fromarray(final_mask.astype(np.uint8))
    else:
        mask_image = final_mask
    
    
    # Remove logo using the mask
    logger.info("Removing logo...")
    try:
        result = remove_logo(original_image, mask_image, threshold)
        
        # Save the result
        result.save(os.path.join(output_dir, "logo_removed.png"))
        logger.info(f"Logo removed image saved to {os.path.join(output_dir, 'logo_removed.png')}")
        
        return final_mask, result
    except Exception as e:
        logger.error(f"Error during logo removal: {e}")
    

if __name__ == "__main__":
    # Example usage
    image_path = "test2.png"  # Replace with your image path
    model_path = "best_model.pth"  # Replace with your model path
    main(image_path, model_path)
