# LookBuilderPipeline

## Installation

```bash
export GH_PAT= ## from doppler secrets
git clone https://${GH_PAT}@github.com/Modegen/LookBuilderPipeline.git
cd LookBuilderPipeline
apt-get update && apt-get install -y libgl1 ## should run automatically in install step
pip install -e .
```
## parameters & their affect on generation:

  - `control_mode=4`,  this dictates `openpose` 
  - `controlnet_conditioning_scale=0.3`
    - 0.2 to 0.4 seem ok -- lower for better outpainting, raise if limbs are off the pose image
  - `strength=0.9` strength of prompt on image
    - this is good for most images -- can override mask/pose if too high (0.95), generate cartoon if too low (0.85) \
    - can be lower based on image, 0.7 works for some images
  - `num_inference_steps=20`
    - 20-40 this is good for most images, sometimes more is needed for more detail
  - `guidance_scale=6`
    - 5 is good for most images, lower for more creative results

# SDXL
### One can run the pipeline with a single input image and a prompt. Mask and pose are optional they will be generated if not provided.
## Running single image generation
```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_sdxl.py \
--image_path=LookBuilderPipeline/img/orig/p20.jpg \
--prompt='a photo realistic image of a fashion model on a beach'

```
# OpenFLUX w/ quant
## Running single image generation
```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_openfl2.py \
--image_path=LookBuilderPipeline/img/orig/p20.jpg \
--prompt='a photo realistic image of a fashion model on a beach'
--quantize='qfloat8'

```

## Running the tests

```bash
python -m pytest
```

## Running the pipeline in python (ipynb)

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

