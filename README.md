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
# from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux


# Initialize the model
model = ImageModelFlux(
    image="path/to/original_image.jpg",
    pose="path/to/pose_image.jpg",
    mask="path/to/mask_image.png",
    prompt="Your text prompt here"
)

# Generate the image
model.load_image()
model.load_model()
generated_image = model.run_model()

# Save or display the generated image
generated_image.save("generated_output_image.png")
generated_image.show()

```

## Evaluate the pipeline

```python
from .LookBuilderPipeline.resize import resize_images
from .LookBuilderPipeline.segment import segment_image

mask,mask_image,mask_array = segment_image('generated_output_image.png',inverse=True,additional_option='shoe')
re_mask=resize_images(mask_image,mask_image.size,aspect_ratio=mask_image.size[0]/mask_image.size[1])

print('mask shape is same:',np.array(re_mask).shape==mask_array.shape)
cov=(np.sum(np.array(re_mask)==mask_array))/np.array(re_mask).size
print('orignal mask coverage to new mask is:',np.array(re_mask).shape==mask_array.shape)

```

## ~~Running the pipeline via RESTFUL API~~ (deprecated)
this needs to be run from the root of the repo
```bash
python3 base_image_model_api.py  
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

