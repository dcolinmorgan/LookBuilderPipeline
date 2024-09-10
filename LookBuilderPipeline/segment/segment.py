from diffusers.utils import load_image
from transformers import pipeline
import numpy as np
from PIL import Image

segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_image(image_path, additional_option=None, resize=False, size=(512,512)):
    """
    Function for segmenting an image and returning the outfit with optional additions.
    Args:
        image (PIL.Image or similar): The actual image object to be processed.
        additional_option (str): The additional item to segment (e.g., "shoes", "handbag"). Optional.
    Returns:
        tuple: (segmented_outfit, mask)
        segmented_outfit (object): The image of the segmented outfit.
        mask (object): A binary mask highlighting the segmented area (outfit and any additional items).
    """
    seg_img = Image.open(image_path)
    segments = segmenter(seg_img)

    segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf"]
    if additional_option in ["shoe"]:
        segment_include.extend(["Left-shoe", "Right-shoe"])
    if additional_option in ["bag"]:
        segment_include.extend(["Bag"])

    mask_list = [np.array(s['mask']) for s in segments if s['label'] in segment_include]

    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
    
    final_array = final_mask.copy()
    final_mask = Image.fromarray(final_mask)
    seg_img.putalpha(final_mask)
    
    if resize==True:
        seg_img = seg_img.resize(size)
        final_mask = final_mask.resize(size)

    return seg_img, final_mask, final_array
