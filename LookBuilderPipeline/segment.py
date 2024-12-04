# Import necessary libraries for image processing and segmentation
from diffusers.utils import load_image  # For loading images
from transformers import pipeline  # For using pre-trained LookBuilderPipeline.models
import numpy as np  # For numerical operations on arrays
from PIL import Image, ImageOps  # For image manipulation
import io
import torch

# Determine the device to use: CUDA, MPS, or CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')  # Use MPS if available
elif torch.cuda.is_available():
    device = torch.device('cuda')  # Use CUDA if available
else:
    device = torch.device('cpu')
# Initialize the segmentation model using a pre-trained model from Hugging Face
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device=device)

def segment_image(image_path, additional_option=None, inverse=True):
    """
    Function for segmenting an image and returning the outfit with optional additions.
    
    Args:
        image_path (str): The path to the image file to be processed.
        additional_option (str): The additional item to segment (e.g., "shoes", "handbag"). Optional.
        
    Returns:
        tuple: (segmented_outfit, mask, final_array)
        segmented_outfit (PIL.Image): The image of the segmented outfit.
        mask (PIL.Image): A binary mask highlighting the segmented area (outfit and any additional items).
        final_array (numpy.ndarray): The final mask as a numpy array.
    """
    # Load the image from the specified path
    if isinstance(image_path,str):
        image = load_image(image_path).convert("RGB")
    elif isinstance(image_path,bytes):
        image = Image.open(io.BytesIO(image_path)).convert("RGB")
    else:
        image=image_path
    
    # Use the segmenter to get segments from the image
    segments = segmenter(image)

    # Define the labels for the segments we want to include
    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf","Left-shoe", "Right-shoe"]
    
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
    image.putalpha(final_mask)
    
    # Return the segmented outfit image, the mask, and the final mask array
    return image, final_mask# final_array

def full_mask(self,original_image,flux_image):
    seg_img = load_image(original_image)
    segments = segmenter(seg_img)
    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf","Left-shoe","Right-shoe"]
    
    seg_img = load_image(flux_image)
    segments = segmenter(seg_img)
    segment_include += [
    "Hat","Hair","Sunglasses","Face","Left-leg","Right-leg","Left-arm","Right-arm"]
    
    
    mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
    
    final_array = final_mask.copy()
    final_mask = Image.fromarray(final_mask)
    # seg_img.putalpha(final_mask)
    
    mask = Image.new("L", final_mask.size)
    mask.paste(final_mask)
    mask = ImageOps.invert(mask)
    final_maskA = mask.point(lambda p: p > 128 and 255)
    # mask_blurred = self.pipe.mask_processor.blur(final_maskA, blur_factor=20)
        
    return final_mask#, mask_blurred



def no_back(self,original_image):
    seg_img = load_image(original_image)
    segments = segmenter(seg_img)
    segment_include = ["Background"]
        
    mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
        
    final_array = final_mask.copy()
    final_mask = Image.fromarray(final_mask)
    seg_img.putalpha(final_mask)
    
    sm_image_A = seg_img.convert("RGBA")

    pixdata = sm_image_A.load()
    
    width, height = sm_image_A.size
    for y in range(height):
        for x in range(width):
            # Change transparent pixels to white
            if pixdata[x, y][3] == 0:  # Check if the alpha value is 0 (transparent)
                pixdata[x, y] = (255, 255, 255, 255)  # Set to white
        
    return seg_img, seg_img_A
