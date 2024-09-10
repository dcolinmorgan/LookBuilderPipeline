from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

def detect_pose(image_path, resize=False):
    """
    function for detecting the pose in an image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        PIL image indicating a pose detection.
    """
    image = load_image(image_path).convert("RGB")
    pose_img = openpose(image)
    if resize:
        pose_img = pose_img.resize((512, 512))
    return pose_img
