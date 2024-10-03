# Import necessary libraries for image processing and segmentation
from diffusers.utils import load_image  # For loading images
from transformers import pipeline  # For using pre-trained models
import numpy as np  # For numerical operations on arrays
from PIL import Image  # For image manipulation
from .resize import resize_images
import cv2

# Initialize the segmentation model using a pre-trained model from Hugging Face
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_image(image_path, additional_option=None, resize=False, size=(512, 512), inverse=False):
    """
    Function for segmenting an image and returning the outfit with optional additions.
    
    Args:
        image_path (str): The path to the image file to be processed.
        additional_option (str): The additional item to segment (e.g., "shoes", "handbag"). Optional.
        resize (bool): Whether to resize the output image. Default is False.
        size (tuple): The target size for resizing the output image. Default is (512, 512).
        
    Returns:
        tuple: (segmented_outfit, mask, final_array)
        segmented_outfit (PIL.Image): The image of the segmented outfit.
        mask (PIL.Image): A binary mask highlighting the segmented area (outfit and any additional items).
        final_array (numpy.ndarray): The final mask as a numpy array.
    """
    # Load the image from the specified path
    seg_img = Image.open(image_path)
    
    # Use the segmenter to get segments from the image
    segments = segmenter(seg_img)

    # Define the labels for the segments we want to include
    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf"]
    
    # Extend the segments to include additional options if specified
    if additional_option in ["shoe"]:
        segment_include.extend(["Left-shoe", "Right-shoe"])
    if additional_option in ["bag"]:
        segment_include.extend(["Bag"])

    # Create a list of masks for the included segments
    if inverse:
        mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
    else:
        mask_list = [np.array(s['mask']) for s in segments if s['label'] in segment_include]

    # Initialize the final mask with the first mask in the list
    final_mask = np.array(mask_list[0])
    
    # Combine all masks into a single final mask
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask  # Add the current mask to the final mask
    
    # Create a copy of the final mask for later use
    final_array = final_mask.copy()
    
    # Convert the final mask to a PIL Image
    final_mask = Image.fromarray(final_mask)
    
    # Add the mask as an alpha channel to the original image
    seg_img.putalpha(final_mask)
    
    # Resize the images if the resize flag is set to True
    if resize:
        seg_img,final_mask = resize_images([seg_img,final_mask], size)

    # Return the segmented outfit image, the mask, and the final mask array
    return seg_img, final_mask, final_array
