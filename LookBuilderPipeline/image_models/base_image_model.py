# Base class for image generation models (SD3, Flux, etc.)
# This contains the shared logic for handling the inputs: pose, clothes, mask, and prompt.

class BaseImageModel:
    def __init__(self, pose, mask, prompt):
        """
        Initialize the image model with common inputs.
        
        Args:
            pose (object): The detected pose generated earlier in the pipeline.
            clothes (object): The segmented clothes (outfit) generated earlier.
            mask (object): The mask generated earlier that defines the boundaries of the outfit.
            prompt (str): The text prompt to guide the image generation (e.g., style or additional details).
        """
        self.pose = pose
        # self.clothes = clothes
        self.mask = mask
        self.prompt = prompt

    def generate_image_extras(image):
        """
        used to generate extra images for the various controlnets
        """
        
        from diffusers.utils import load_image
        from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
        label=str(np.random.randint(100000000))
        
        #pose
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        pose_image = openpose(image)

        #mask
        segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")
        segments = segmenter(image)
        segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf", "Right-shoe","Left-shoe","Bag"]
        mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
        final_mask = np.array(mask_list[0])
        for mask in mask_list:
            current_mask = np.array(mask)
            final_mask = final_mask + current_mask  # Add the current mask to the final mask
        final_array = final_mask.copy() 
        final_mask = Image.fromarray(final_mask)

        #canny (if needed)
        canny = CannyDetector()
        canny_image = canny(image)


        showImagesHorizontally([image,final_mask,canny_image,pose_image],'input'+label+'.png')

        return final_mask,canny_image,pose_image
