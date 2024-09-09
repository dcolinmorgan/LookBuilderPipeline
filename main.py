# Import the necessary modules from your project

from segmentation.segmentation import segment_image
from pose.pose import detect_pose
from image_models.image_model_sd3 import generate_image_sd3
from image_models.image_model_fl2 import generate_image_flux


# Function to run the pipeline with the Stable Diffusion 3 model
def run_pipeline_sd3(image, prompt):
    """
    Run the entire pipeline and generate the final image using Stable Diffusion 3 with ControlNet.
    
    Args:
        image (object): The input image object.
        prompt (str): The textual prompt to guide the image generation.
    
    Returns:
        object: The generated image using Stable Diffusion 3.
    """
    # Step 1: Segment the image to extract the outfit (minimum) and additional elements
    segmented_clothes, mask = segment_image(image)
    print("Segmentation completed.")

    # Step 2: Detect the pose of the model in the image
    pose = detect_pose(image)
    print("Pose detection completed.")

    # Step 3: Generate the final image using Stable Diffusion 3
    generated_image = generate_image_sd3(pose, segmented_clothes, mask, prompt)
    print("Image generation with SD3 completed.")
    
    return generated_image


# Function to run the pipeline with the Flux model
def run_pipeline_flux(image, prompt):
    """
    Run the entire pipeline and generate the final image using the Flux model.
    
    Args:
        image (object): The input image object.
        prompt (str): The textual prompt to guide the image generation.
    
    Returns:
        object: The generated image using the Flux model.
    """
    # Step 1: Segment the image to extract the outfit (minimum) and additional elements
    segmented_clothes, mask = segment_image(image)
    print("Segmentation completed.")

    # Step 2: Detect the pose of the model in the image
    pose = detect_pose(image)
    print("Pose detection completed.")

    # Step 3: Generate the final image using the Flux model
    generated_image = generate_image_flux(pose, segmented_clothes, mask, prompt)
    print("Image generation with Flux completed.")
    
    return generated_image


# Main execution for testing both pipelines
if __name__ == "__main__":
    # Placeholder image and prompt (replace these with actual image and prompt)
    test_image = "sample_image.jpg"  # Replace with actual image object
    test_prompt = "A model in a futuristic outfit"  # Example prompt

    # Run the pipeline with Stable Diffusion 3
    print("Running pipeline with Stable Diffusion 3...")
    sd3_output = run_pipeline_sd3(test_image, test_prompt)
    print(f"Output from SD3 model: {sd3_output}")

    # Run the pipeline with the Flux model
    print("Running pipeline with Flux model...")
    flux_output = run_pipeline_flux(test_image, test_prompt)
    print(f"Output from Flux model: {flux_output}")
