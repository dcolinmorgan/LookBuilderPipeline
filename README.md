# LookBuilderPipeline

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Running the tests

```bash
python -m pytest
```

# One can run the pipeline with a single input image and a prompt. Mask and pose are optional they will be generated if not provided.
## Running single image generation
```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_sdxl.py \
--image_path=LookBuilderPipeline/img/orig/p20.jpg \
--prompt='a photo realistic image of a fashion model on a beach'

```
## Running the pipeline

```python
from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
# or from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux

def run_pipeline():
    # Initialize the model
    model = ImageModelSDXL(
        pose="path/to/pose_image.jpg",
        mask="path/to/mask_image.png",
        prompt="Your text prompt here"
    )
    
    # Generate the image
    generated_image = model.generate_image()
    
    # Save or display the generated image
    generated_image.save("output_image.png")
    generated_image.show()

if __name__ == "__main__":
    run_pipeline()
```

## Running the pipeline via API (marked for deletion)
this needs to be run from the root of the repo
```bash
python3 base_image_model_api.py  # not pushed -- delete? 
```
```bash
   curl -X POST \
     -F "original_image=@path/to/original_image.jpg" \
     -F "pose_image=@path/to/pose_image.jpg" \
     -F "mask_image=@path/to/mask_image.png" \
     -F "model=sdxl" \
     -F "prompt=your text prompt here" \
     http://localhost:5005/generate_image
```

