import argparse
import torch
import wandb
import os

from models import LogoCleanerModel
from train import train_logo_cleaner
from inference import main as run_inference
from dataloader import create_logo_dataloader

def main():
    parser = argparse.ArgumentParser(description="Logo Cleaner Pipeline with SlimSAM")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    train_parser.add_argument("--wandb-project", type=str, default="logo-cleaner", help="Weights & Biases project name")
    train_parser.add_argument("--wandb-run", type=str, default="slimSAM-training", help="Weights & Biases run name")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument("--image", type=str, required=True, help="Path to input image")
    infer_parser.add_argument("--model", type=str, default="best_model.pth", help="Path to trained model")
    infer_parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (0-1)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.command == "train":
        # Create checkpoint directory if it doesn't exist
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run)
        
        # Since the user mentioned they already have the dataloader set up,
        # we expect them to customize this part
        
        # Example of how the dataloader might be used:
        train_dataloader = create_logo_dataloader(
            root_dir='/mnt1/peter/datasets', 
            filelist_path='/mnt1/peter/datasets/train/filelist-logosonly.txt',
        )
        train_logo_cleaner(
            train_dataloader,
            val_dataloader=None,  # Replace with actual validation dataloader if available
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device
        )
        
    elif args.command == "infer":
        # Run inference
        run_inference(
            image_path=args.image,
            model_path=args.model,
            threshold=args.threshold
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
    # For training
    # python main.py train --epochs 50 --lr 1e-4

    # # For inference
    # python main.py infer --image test2.png --model best_model.pth