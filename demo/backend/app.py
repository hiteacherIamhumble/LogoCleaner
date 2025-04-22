from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
import json
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io
import base64
import sys
import logging
import traceback
import threading

# Set up logging and increase timeout for OpenAI API calls
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Flask to handle longer requests
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the functions from backend.py
try:
    from backend import load_model, preprocess_image, predict_and_refine_with_sam, visualize_and_save_results, remove_logo, logo_detection, generate_image
    logger.info("Successfully imported backend.py functions")
    USE_ML_BACKEND = True
except ImportError as e:
    logger.error(f"Failed to import from backend.py: {e}")
    USE_ML_BACKEND = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
RESULTS_FOLDER = os.path.join(current_dir, 'results')
EXAMPLE_FOLDER = os.path.join(current_dir, 'examples')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(EXAMPLE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['EXAMPLE_FOLDER'] = EXAMPLE_FOLDER

# Global variables for model
model = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resolve_file_path(filepath):
    """Try to resolve a file path, checking different possible locations."""
    # First check if the path exists as is
    if os.path.exists(filepath):
        return filepath
        
    # If it's not an absolute path, try to find it in the uploads folder
    filename = os.path.basename(filepath)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(upload_path):
        logger.info(f"Found file in uploads folder: {upload_path}")
        return upload_path
    
    # Try just the filename in the current directory
    cwd_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(cwd_path):
        logger.info(f"Found file in current directory: {cwd_path}")
        return cwd_path
    
    # If we can't find it, log debug info and return None
    logger.warning(f"Could not resolve file path: {filepath}")
    logger.warning(f"Current directory: {os.getcwd()}")
    logger.warning(f"Files in uploads directory: {os.listdir(UPLOAD_FOLDER)}")
    return None

# Fallback function for when ML backend is not available
def dummy_process_image(image_path, output_dir):
    """Simple function to process images without ML models"""
    # try:
    #     # Open the original image
    #     original_image = Image.open(image_path).convert("RGB")
        
    #     # Save the original image
    #     original_image.save(os.path.join(output_dir, "original.png"))
        
    #     # Create a copy with a green "bounding box" for demonstration
    #     bbox_image = original_image.copy()
    #     width, height = bbox_image.size
    #     # Draw a simple green rectangle in the center (30% of the image)
    #     bbox = (
    #         int(width * 0.35), 
    #         int(height * 0.35), 
    #         int(width * 0.65), 
    #         int(height * 0.65)
    #     )
        
    #     # Draw rectangle on the image
    #     from PIL import ImageDraw
    #     draw = ImageDraw.Draw(bbox_image)
    #     draw.rectangle(bbox, outline="green", width=3)
    #     bbox_image.save(os.path.join(output_dir, "original_with_bbox.png"))
        
    #     # Create a "mask" overlay (red tint in the center)
    #     mask_overlay = original_image.copy()
    #     mask = Image.new('L', original_image.size, 0)
    #     mask_draw = ImageDraw.Draw(mask)
    #     mask_draw.rectangle(bbox, fill=128)
        
    #     # Apply red tint to the masked area
    #     mask_array = np.array(mask_overlay)
    #     mask_array[(np.array(mask) > 64)] = mask_array[(np.array(mask) > 64)] * 0.7 + np.array([255, 0, 0]) * 0.3
    #     mask_overlay = Image.fromarray(mask_array.astype(np.uint8))
    #     mask_overlay.save(os.path.join(output_dir, "refined_mask_overlay.png"))
        
    #     # Create a "logo removed" version (just blur the center area)
    #     removed = original_image.copy()
    #     mask_blur = original_image.crop(bbox).resize(
    #         (bbox[2] - bbox[0], bbox[3] - bbox[1]), 
    #         Image.LANCZOS
    #     ).filter(Image.BoxBlur(10))
        
    #     removed.paste(mask_blur, bbox)
    #     removed.save(os.path.join(output_dir, "logo_removed.png"))
        
    #     return True
    # except Exception as e:
    #     logger.error(f"Error in dummy processing: {e}")
    #     logger.error(traceback.format_exc())
    #     return False

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'ml_backend': USE_ML_BACKEND}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("Upload endpoint called")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    logger.info(f"Request files: {request.files}")
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"File saved at {file_path}")
        
        # Validate file exists after saving
        if not os.path.exists(file_path):
            logger.error(f"File not found after saving: {file_path}")
            return jsonify({'error': 'File not found after saving'}), 500
        
        # Return the file path for the next steps
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': file_path
        }), 200
    
    logger.warning(f"File type not allowed: {file.filename}")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/create-job', methods=['POST'])
def create_job():
    logger.info("Create job endpoint called")
    data = request.json
    logger.info(f"Request data: {data}")
    
    filepath = data.get('filepath')
    
    # More detailed logging for filepath issues
    if not filepath:
        logger.warning("No filepath provided in request")
        return jsonify({'error': 'No filepath provided'}), 400
    
    # Resolve the file path
    resolved_path = resolve_file_path(filepath)
    if not resolved_path:
        return jsonify({'error': 'File not found at the specified path'}), 400
    
    # Update the filepath with the resolved path
    filepath = resolved_path
    logger.info(f"Resolved file path: {filepath}")
    
    try:
        # Create a unique ID for this processing job
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        
        # Initialize job info
        job_info = {
            'filepath': filepath,
            'processing_stage': 'initialized'
        }
        
        # Save original image immediately so it can be shown to the user
        try:
            # Copy the original image
            original_image = Image.open(filepath).convert("RGB")
            original_image.save(os.path.join(output_dir, "original.png"))
            
            # Save job info
            with open(os.path.join(output_dir, "job_info.json"), 'w') as f:
                json.dump(job_info, f)
            
            # Don't create any other image placeholders - we'll only show the original at first
            
        except Exception as e:
            logger.error(f"Error saving original image: {e}")
            # Only create a placeholder for the original image if needed
            placeholder = Image.new('RGB', (512, 512), (240, 240, 240))
            placeholder.save(os.path.join(output_dir, "original.png"))
        
        # Prepare response with only the original image path
        response = {
            'job_id': job_id,
            'status': 'initialized',
            'results': {
                'original': f'/api/results/{job_id}/original.png',
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-process', methods=['POST'])
def start_process_image():
    global model, device
    
    logger.info("Start processing endpoint called")
    data = request.json
    logger.info(f"Request data: {data}")
    
    filepath = data.get('filepath')
    
    # More detailed logging for filepath issues
    if not filepath:
        logger.warning("No filepath provided in request")
        return jsonify({'error': 'No filepath provided'}), 400
    
    # Resolve the file path
    resolved_path = resolve_file_path(filepath)
    if not resolved_path:
        return jsonify({'error': 'File not found at the specified path'}), 400
    
    # Update the filepath with the resolved path
    filepath = resolved_path
    logger.info(f"Resolved file path: {filepath}")
    
    try:
        # Create a unique ID for this processing job
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        
        # Initialize job info
        job_info = {
            'filepath': filepath,
            'processing_stage': 'initialized'
        }
        
        # Save original image immediately so it can be shown to the user
        try:
            # Copy the original image
            original_image = Image.open(filepath).convert("RGB")
            original_image.save(os.path.join(output_dir, "original.png"))
            
            # Save job info
            with open(os.path.join(output_dir, "job_info.json"), 'w') as f:
                json.dump(job_info, f)
                
            # Process the image to detect logo
            if USE_ML_BACKEND:
                try:
                    # Initialize model if not done yet
                    if model is None:
                        try:
                            import torch
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            logger.info(f"Using device: {device}")
                            
                            # Try to load model, fall back to dummy mode on failure
                            model_path = os.path.join(current_dir, "best_model.pth")
                            if os.path.exists(model_path):
                                logger.info(f"Loading model from {model_path}")
                                model = load_model(model_path, device)
                            else:
                                logger.warning(f"Model file not found at {model_path}, using dummy mode")
                                raise FileNotFoundError(f"Model file not found at {model_path}")
                        except Exception as e:
                            logger.error(f"Error initializing model: {e}")
                            raise
                    
                    # Detect logo using the ML model
                    success, _ = logo_detection(model, filepath, output_dir, device)
                    
                    if success:
                        # Update job info
                        job_info['processing_stage'] = 'detected'
                        with open(os.path.join(output_dir, "job_info.json"), 'w') as f:
                            json.dump(job_info, f)
                    else:
                        # Fall back to dummy processing
                        dummy_process_image(filepath, output_dir)
                
                except Exception as e:
                    logger.error(f"Error in logo detection: {e}")
                    logger.error(traceback.format_exc())
                    # Create placeholders for the other images
                    placeholder = Image.new('RGB', (512, 512), (240, 240, 240))
                    placeholder.save(os.path.join(output_dir, "original_with_bbox.png"))
                    placeholder.save(os.path.join(output_dir, "refined_mask_overlay.png"))
                    placeholder.save(os.path.join(output_dir, "logo_removed.png"))
            else:
                # Use dummy processing if ML backend is not available
                logger.info("Using dummy processing mode")
                dummy_process_image(filepath, output_dir)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Create placeholders for all images
            placeholder = Image.new('RGB', (512, 512), (240, 240, 240))
            placeholder.save(os.path.join(output_dir, "original.png"))
            placeholder.save(os.path.join(output_dir, "original_with_bbox.png"))
            placeholder.save(os.path.join(output_dir, "refined_mask_overlay.png"))
            placeholder.save(os.path.join(output_dir, "logo_removed.png"))
        
        # Ensure all required files exist
        required_files = ["original.png", "original_with_bbox.png", "refined_mask_overlay.png"]
        for filename in required_files:
            file_path = os.path.join(output_dir, filename)
            if not os.path.exists(file_path):
                logger.warning(f"Missing required output file: {filename}, creating placeholder")
                placeholder = Image.new('RGB', (512, 512), (240, 240, 240))
                placeholder.save(file_path)
        
        # Prepare response with image paths
        response = {
            'job_id': job_id,
            'status': 'processing',
            'results': {
                'original': f'/api/results/{job_id}/original.png',
                'bbox': f'/api/results/{job_id}/original_with_bbox.png',
                'mask': f'/api/results/{job_id}/refined_mask_overlay.png',
                'removed': f'/api/results/{job_id}/logo_removed.png'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/<job_id>', methods=['POST', 'OPTIONS'])
def generate_final_image(job_id):
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    global model, device
    
    logger.info(f"Generate image endpoint called for job_id: {job_id}")
    
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory does not exist: {output_dir}")
        return jsonify({'error': 'Invalid job ID'}), 400
    
    # Load job info
    try:
        with open(os.path.join(output_dir, "job_info.json"), 'r') as f:
            job_info = json.load(f)
        
        filepath = job_info.get('filepath')
        processing_stage = job_info.get('processing_stage')
        
        if not filepath:
            logger.warning("No filepath provided in job info")
            return jsonify({'error': 'No filepath in job info'}), 400
        
        # Resolve the file path
        resolved_path = resolve_file_path(filepath)
        if not resolved_path:
            return jsonify({'error': 'Original file not found at the specified path'}), 400
        
        # Update the filepath with the resolved path
        filepath = resolved_path
        logger.info(f"Resolved file path: {filepath}")
        
        # Check if already fully processed
        if processing_stage == 'completed':
            logger.info(f"Job {job_id} already processed")
            return jsonify({
                'job_id': job_id,
                'status': 'success',
                'results': {
                    'original': f'/api/results/{job_id}/original.png',
                    'bbox': f'/api/results/{job_id}/original_with_bbox.png',
                    'mask': f'/api/results/{job_id}/refined_mask_overlay.png',
                    'removed': f'/api/results/{job_id}/logo_removed.png'
                }
            }), 200
            
        # Generate the logo-removed image
        if USE_ML_BACKEND:
            try:
                # Initialize model if not done yet
                if model is None:
                    try:
                        import torch
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        logger.info(f"Using device: {device}")
                        
                        # Try to load model, fall back to dummy mode on failure
                        model_path = os.path.join(current_dir, "best_model.pth")
                        if os.path.exists(model_path):
                            logger.info(f"Loading model from {model_path}")
                            model = load_model(model_path, device)
                        else:
                            logger.warning(f"Model file not found at {model_path}, using dummy mode")
                            raise FileNotFoundError(f"Model file not found at {model_path}")
                    except Exception as e:
                        logger.error(f"Error initializing model: {e}")
                        raise
                
                # Process the image using ML backend
                logger.info(f"Generating final image for job {job_id}")
                
                # Generate and save result
                success = generate_image(model, filepath, output_dir, device)
                if success:
                    # Update job info
                    job_info['processing_stage'] = 'completed'
                    with open(os.path.join(output_dir, "job_info.json"), 'w') as f:
                        json.dump(job_info, f)
                    
                    logger.info(f"Successfully generated final image for job {job_id}")
                else:
                    logger.warning(f"Failed to generate final image for job {job_id}")
                    # Fall back to dummy processing
                    dummy_process_image(filepath, output_dir)
                    
            except Exception as e:
                logger.error(f"Error in ML processing: {e}")
                logger.error(traceback.format_exc())
                # Fall back to dummy processing
                dummy_process_image(filepath, output_dir)
        else:
            # Use dummy processing if ML backend is not available
            logger.info("Using dummy processing mode")
            dummy_process_image(filepath, output_dir)
        
        # Ensure the logo_removed.png file exists
        removed_path = os.path.join(output_dir, "logo_removed.png")
        if not os.path.exists(removed_path):
            logger.warning("Missing logo_removed.png, creating placeholder")
            placeholder = Image.new('RGB', (512, 512), (240, 240, 240))
            placeholder.save(removed_path)
        
        # Prepare response with image paths
        response = {
            'job_id': job_id,
            'status': 'success',
            'results': {
                'original': f'/api/results/{job_id}/original.png',
                'bbox': f'/api/results/{job_id}/original_with_bbox.png',
                'mask': f'/api/results/{job_id}/refined_mask_overlay.png',
                'removed': f'/api/results/{job_id}/logo_removed.png'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/<filename>', methods=['GET'])
def get_result(job_id, filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], job_id, filename)
    logger.info(f"Requesting result file: {filepath}")
    
    # Get the cache-busting timestamp if present
    timestamp = request.args.get('t', '')
    
    if os.path.exists(filepath):
        # Check if it's an empty or very small file (could be a placeholder)
        file_size = os.path.getsize(filepath)
        if file_size < 100:  # If file is suspiciously small
            logger.warning(f"File exists but is too small: {filepath} ({file_size} bytes)")
            # Return a transparent placeholder
            placeholder = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
            img_io = io.BytesIO()
            placeholder.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        
        # Add cache control headers to prevent browser caching
        response = send_file(filepath, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        logger.warning(f"File not found: {filepath}")
        # Return a transparent placeholder image instead of 404
        placeholder = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        img_io = io.BytesIO()
        placeholder.save(img_io, 'PNG')
        img_io.seek(0)
        response = send_file(img_io, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

@app.route('/api/examples', methods=['GET'])
def get_examples():
    logger.info("Examples endpoint called")
    
    # Create some example files if directory is empty
    if not os.listdir(app.config['EXAMPLE_FOLDER']):
        logger.info("Creating example files")
        try:
            # Create sample examples
            for i in range(1, 4):
                example = Image.new('RGB', (512, 512), (200, 200, 200))
                example.save(os.path.join(app.config['EXAMPLE_FOLDER'], f"example{i}.png"))
        except Exception as e:
            logger.error(f"Error creating example files: {e}")
    
    examples = []
    for filename in os.listdir(app.config['EXAMPLE_FOLDER']):
        if allowed_file(filename):
            example_path = os.path.join(app.config['EXAMPLE_FOLDER'], filename)
            examples.append({
                'id': filename,
                'name': filename,
                'url': f'/api/example/{filename}'
            })
    
    logger.info(f"Returning {len(examples)} examples")
    return jsonify(examples), 200

@app.route('/api/example/<filename>', methods=['GET'])
def get_example(filename):
    filepath = os.path.join(app.config['EXAMPLE_FOLDER'], filename)
    logger.info(f"Requesting example file: {filepath}")
    
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        logger.warning(f"Example not found: {filepath}")
        # Return a placeholder image
        placeholder = Image.new('RGB', (512, 512), (200, 200, 200))
        img_io = io.BytesIO()
        placeholder.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    # Increase the timeout for the server
    app.run(debug=True, host='0.0.0.0', port=5050, threaded=True, request_handler=WSGIRequestHandler)
