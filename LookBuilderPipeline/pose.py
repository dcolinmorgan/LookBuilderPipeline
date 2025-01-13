# Import necessary libraries for pose detection

from diffusers.utils import load_image  # For loading images
import os,sys
import torch
from PIL import Image
import io

from .DWPose.src.dwpose import DwposeDetector

class PoseDetector:
    def __init__(self):
        self.model = DwposeDetector.from_pretrained_default()
        self._setup_device()
        self._block_print()
    
    def _setup_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def _block_print(self):
        sys.stdout = open(os.devnull, 'w')
    
    def detect_pose(self, image_path, face=True):
        """Process image and detect pose."""
        image = self._load_image(image_path)
        return self._process_pose(image, face)
    
    def _load_image(self, image_path):
        if isinstance(image_path, str):
            return load_image(image_path).convert("RGB")
        elif isinstance(image_path, bytes):
            return Image.open(io.BytesIO(image_path)).convert("RGB")
        return image_path
    
    def _process_pose(self, image, face):
        pose_img, _, _ = self.model(
            image,
            include_hand=True,
            include_face=face,
            include_body=True,
            include_foot=True,
            image_and_json=True,
            detect_resolution=512,
            device=self.device
        )
        return pose_img
