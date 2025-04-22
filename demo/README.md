# LogoCleaner - AI-Powered Logo Removal Tool

LogoCleaner is a web application that uses AI to detect and remove logos from images. The application consists of a Flask backend for image processing and a Vue.js frontend for the user interface.

## Features

- Upload images for logo removal
- AI-powered logo detection
- Precise logo segmentation using SAM (Segment Anything Model)
- Background filling with OpenAI's image generation
- Step-by-step visualization of the logo removal process
- Gallery of example images

## Project Structure

```
logo-cleaner/
│
├── backend/              # Flask backend
│   ├── app.py            # Main Flask application
│   ├── backend.py        # Logo removal logic
│   ├── models.py         # AI models
│   └── requirements.txt  # Python dependencies
│
└── frontend/            # Vue.js frontend
    ├── public/          # Static assets
    ├── src/             # Vue source code
    │   ├── components/  # Reusable components
    │   ├── views/       # Page components
    │   ├── assets/      # Images, fonts, etc.
    │   ├── router/      # Vue Router configuration
    │   ├── App.vue      # Main application component
    │   └── main.js      # Application entry point
    ├── package.json     # NPM dependencies
    └── vue.config.js    # Vue configuration
```

## Hardware Requirements

- CPU-only machine is sufficient for running the demo application
- No GPU required for inference

## Prerequisites

- Python 3.8+
- Node.js 14+
- PyTorch 1.9+ (CPU version)
- OpenAI API key (optional for enhanced inpainting)

## Environment Setup with Conda

1. Create and activate a new conda environment:
   ```bash
   conda create -n logocleaner python=3.8
   conda activate logocleaner
   ```

2. Install PyTorch (CPU version):
   ```bash
   conda install pytorch torchvision cpuonly -c pytorch
   ```

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the SAM model checkpoint (if not already present):
   ### Direct Download
    - Model link: [LogoCleaner Model on HuggingFace](https://huggingface.co/PeterDAI/LogoCleaner/tree/main)
    - Place the downloaded model in the current directory.

   ### Using the Hugging Face Hub

   ```python
   from huggingface_hub import hf_hub_download

   # Download the SAM model
   sam_path = hf_hub_download(
         repo_id="PeterDAI/LogoCleaner",
         filename="sam_vit_b_01ec64.pth"
   )

   # Download the selector model
   selector_path = hf_hub_download(
         repo_id="PeterDAI/LogoCleaner",
         filename="best_model.pth"
   )
   ```

4. Set your OpenAI API key (optional for enhanced inpainting, Our team already place the key inside and you can directly use it):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

5. Run the Flask application:
   ```bash
   python app.py
   ```
   The backend will run at http://localhost:5050

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will be accessible at http://localhost:8080


## Usage

1. Open your browser and navigate to http://localhost:8080
2. Upload an image that contains a logo you want to remove
3. Click the "Remove Logo" button
4. Wait for the AI to detect and remove the logo
5. Download the result or process another image

## Troubleshooting

- If you encounter issues with the backend, check that all dependencies are correctly installed
- For frontend issues, make sure Node.js version is compatible (v14+)
- Ensure both backend and frontend servers are running simultaneously
- The application can work in demo mode even without ML models by using placeholder images

## License

This project is licensed under the MIT License - see the LICENSE file for details.
