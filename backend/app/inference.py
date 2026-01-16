import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2  # OpenCV: Used for image artifact removal
from . import networks
import os


class SunPredictor:
    def __init__(self, model_path):
        # 1. Automatically detect device (Use GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing model structure from networks.py...")
        # 2. Define the generator structure (Must match training config: unet_256)
        # Key Point A: use_dropout=False -> Disable random dropout to reduce noise artifacts at source
        self.netG = networks.define_G(input_nc=3, output_nc=3, ngf=64,
                                      netG='unet_256', norm='batch',
                                      use_dropout=False, init_type='normal',
                                      init_gain=0.02)

        # 3. Load model weights
        self.load_weights(model_path)
        self.netG.to(self.device)

    def load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        # Load the state dictionary containing the trained parameters
        state_dict = torch.load(path, map_location=self.device)

        # Clean up potential metadata keys in the dictionary to prevent errors
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.netG.load_state_dict(state_dict)

    def preprocess(self, image: Image.Image):
        """
        Convert human-readable image to AI-readable Tensor (Range: -1 to 1).
        """
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Force resize to fixed input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # unsqueeze(0) adds a dimension to represent batch_size = 1
        return transform(image).unsqueeze(0).to(self.device)

    def postprocess_and_fix(self, tensor):
        """
        Convert AI output Tensor back to image and apply 'Engineering Fixes'
        to remove artifacts (Red Dots & Blue Borders).
        """

        # 1. Data Conversion: Tensor -> Numpy
        tensor = tensor.detach().cpu().float()
        tensor = (tensor + 1) / 2.0 * 255.0  # Scale from [-1, 1] back to [0, 255]
        tensor = tensor.clamp(0, 255).byte()  # Ensure values are valid bytes
        tensor = tensor.squeeze(0)
        # PyTorch is (C, H, W), OpenCV needs (H, W, C), so we permute dimensions
        numpy_img = tensor.permute(1, 2, 0).numpy()

        # --- Engineering Fixes (Artifact Removal) ---

        # Fix A: Remove Red Dot Artifacts (Salt-and-pepper noise)
        # Principle: Median Blur (kernel=3) replaces isolated noise pixels
        # with the median value of neighbors, preserving edges.
        numpy_img = cv2.medianBlur(numpy_img, 3)

        # Fix B: Remove Edge Artifacts (Blue lines/Black borders)
        # Principle: Aggressively crop the outer 2 pixels where padding artifacts occur.
        h, w, _ = numpy_img.shape
        mask_top = 29
        mask_bottom = 22
        mask_left = 26
        mask_right = 25

        fill_color = [255, 255, 255]

        if mask_top > 0:
            numpy_img[0:mask_top, :] = fill_color
        if mask_bottom > 0:
            numpy_img[h - mask_bottom:h, :] = fill_color
        if mask_left > 0:
            numpy_img[:, 0:mask_left] = fill_color
        if mask_right > 0:
            numpy_img[:, w - mask_right:w] = fill_color


        # Fix C: Restore Dimensions
        # Since cropping reduced the size, we resize back to 256x256
        # to ensure the texture maps correctly on the frontend.
        numpy_img = cv2.resize(numpy_img, (512, 512), interpolation=cv2.INTER_NEAREST)

        # -----------------------------------------------

        return Image.fromarray(numpy_img)

    def predict(self, image_file):
        """
        Main entry point for external calls:
        Input Image -> Inference -> Artifact Removal -> Output Image
        """
        # Ensure input is RGB (prevents errors with single-channel grayscale inputs)
        img = Image.open(image_file).convert('RGB')

        input_tensor = self.preprocess(img)

        # torch.no_grad() disables gradient calculation to save VRAM and speed up inference
        with torch.no_grad():
            output_tensor = self.netG(input_tensor)

        # Return the clean, processed image
        return self.postprocess_and_fix(output_tensor)