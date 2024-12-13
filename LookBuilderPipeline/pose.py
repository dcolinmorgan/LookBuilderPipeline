# Import necessary libraries for pose detection
# from controlnet_aux import OpenposeDetector  # For pose detection using OpenPose
from diffusers.utils import load_image  # For loading images
import os,sys
import torch
from PIL import Image
import io
# # Initialize the OpenPose detector using a pre-trained model from Hugging Face
# openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
# from controlnet_aux.processor import Processor
# processor_id = 'openpose_full'
# openpose = Processor(processor_id)

from .DWPose.src.dwpose import DwposeDetector

model = DwposeDetector.from_pretrained_default()

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
blockPrint()

def detect_pose(image_path, face=True):
    """
    Function for detecting the pose in an image.
    
    Args:
        image_path (str): Path to the input image.
        face (bool): Whether to include the face in the pose detection.
    Returns:
        PIL.Image: An image indicating the detected pose.
    """
    # Load the image from the specified path and convert it to RGB format
    if isinstance(image_path,str):
        image = load_image(image_path).convert("RGB")
    elif isinstance(image_path,bytes):
        image = Image.open(io.BytesIO(image_path)).convert("RGB")
    else:
        image=image_path

    # Determine the device to use: CUDA, MPS, or CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Use MPS if available
    elif torch.cuda.is_available():
        device = torch.device('cuda')  # Use CUDA if available
    else:
        device = torch.device('cpu')
    # Use the OpenPose detector to detect the pose in the image
    # pose_img = openpose(image) #,include_hand=True,include_face=False)
    
    if image.size[0]>image.size[1]:
        resolution=image.size[0]
    else:
        resolution=image.size[1]
    pose_img,j,source = model(image,
        include_hand=True,
        include_face=face,
        include_body=True,
        include_foot=True,
        image_and_json=True,
        detect_resolution=resolution,
        device=device)
    
    # pose_image = openpose(image, hand_and_face=False, output_type='cv2')

    # Return the image with the detected pose
    return pose_img
