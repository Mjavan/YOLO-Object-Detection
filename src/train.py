import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import ultralytics
from ultralytics import YOLO
from pathlib import Path
import argparse
import os, sys


print("===== ENVIRONMENT =====")
print("Python:", sys.executable)
print("Inside Enroot:", bool(os.environ.get("ENROOT_ROOTFS")))
print("=======================")

#  data_path, model_name='yolov8n.pt', epochs=100, batch_size=16, img_size=640, project='runs/train', name='exp'
class YOLOTrainer:
    def __init__(self,args):
        self.args = args
        self.data_path = self.args.yaml_path
        self.model_name = self.args.model_name
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.img_size = self.args.img_size
        self.project = self.args.project
        self.name = self.args.name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        # Load model
        model = YOLO(self.model_name)
        print("Model loaded successfully.")
        
        print("Starting training...")

        # CPU info
        print("==========CPU Info==========")
        print("Torch threads:", torch.get_num_threads())
        print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
        print("Total CPUs visible:", os.cpu_count())

        print("==========GPU Info==========")

        # Train model
        model.train(data=self.data_path,
                    epochs=self.epochs,
                    batch=self.batch_size,
                    imgsz=self.img_size,
                    project=self.project,
                    mixup=0.0,
                    name=self.name,
                    device=self.device,
                    workers=0,
                    deterministic=True,
                    seed=42)


parser= argparse.ArgumentParser(description="Train YOLOv8 model")
parser.add_argument('--yaml_path', type=str,default= "kitti.yaml", help='Path to the YAML file')
parser.add_argument('--model_name', type=str, default='yolov8s.pt', help='Pre-trained model name')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--img_size', type=int, default=832, help='Image size for training')
parser.add_argument('--project', type=str, default='runs01', help='project name')
parser.add_argument('--name', type=str, default='kitti_exp', help='name')
parser.add_argument('--patince', type=int, default=20, help='Number of epochs with no improvement to stop training')
args = parser.parse_args()

if __name__ == "__main__":
    trainer = YOLOTrainer(args)
    trainer.train()
    print('training is done!')









