# Import necessary libraries for pose detection
from controlnet_aux import OpenposeDetector  # For pose detection using OpenPose
from diffusers.utils import load_image  # For loading images
from .resize import resize_images

# Initialize the OpenPose detector using a pre-trained model from Hugging Face
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

def detect_pose(image_path, resize=False, size=(512, 512)):
    """
    Function for detecting the pose in an image.
    
    Args:
        image_path (str): Path to the input image.
        resize (bool): Whether to resize the output image. Default is False.
        size (tuple): The target size for resizing the output image. Default is (512, 512).
        
    Returns:
        PIL.Image: An image indicating the detected pose.
    """
    # Load the image from the specified path and convert it to RGB format
    image = load_image(image_path).convert("RGB")
    
    # Use the OpenPose detector to detect the pose in the image
    pose_img = openpose(image)
    
    # pose_image = openpose(image, hand_and_face=False, output_type='cv2')
    
    # Resize the pose image if the resize flag is set to True
    if resize:
        pose_img = resize_images(pose_img,size)
        # pose_image = resize_images(pose_image,size)
    
    # Return the image with the detected pose
    return pose_img
