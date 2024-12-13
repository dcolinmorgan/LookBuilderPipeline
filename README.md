# LookBuilderPipeline

## Installation

```bash
export GH_PAT= ## from doppler secrets
git clone -b feature/alpha_plus_sdxl https://${GH_PAT}@github.com/Modegen/LookBuilderPipeline.git
# may need this next step on some INTEL machines
# apt-get update && apt-get install -y libgl1 ## should run automatically in install step
pip install -e LookBuilderPipeline/.
```
## parameters & their affect on generation:

  - `control_mode=4`,  this dictates `openpose` 
  - `controlnet_conditioning_scale=1.0`
    - 0.2 to 0.4 seem ok -- lower for better outpainting, raise if limbs are off the **pose** image
  - `strength=1.0` strength of **mask** on generation
    - this is good for most images -- can override mask/pose if too high (0.95), generate cartoon if too low (0.85) \
    - can be lower based on image, 0.7 works for some images
  - `num_inference_steps=28`
    - 20-40 this is good for most images, sometimes more is needed for more detail
  - `guidance_scale=7.5` **prompt**
    - 5 is good for most images, lower for more creative results

# SDXL
### One can run the pipeline with a single input image and a prompt. Mask and pose are optional they will be generated if not provided.
```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_sdxl.py \
--image_path=tests/img/p09.jpg \
--prompt='a photo realistic image of a fashion model on a beach'
```
# SDXL BENCHMARK
### will run through all images in test-ai/upscaled/ and use the default parameters, unless user specifies, +/- 0.1

```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_sdxl.py \
--image_path=test-ai/bench/* \
--prompt="A supermodel sits elegantly on Luxury hotel pool side with palms at night, skin reflects hotel in the desert surrounded by dark, rugged terrain and towering volcanic peaks. She wears high-fashion clothing, contrasting with the dramatic landscape. Her hair flows gently in the wind as she gazes into the distance, under a moody sky with soft light breaking through the clouds. The scene blends natural beauty with modern glamour, highlighting the model against the striking volcanic background." \
--benchmark='True'
```
### with LoRA
```bash
python3 LookBuilderPipeline/LookBuilderPipeline/image_models/image_model_sdxl.py \
--image_path=shootjpglow-sorted/referance_images/* \
--prompt="A supermodel sits elegantly on Luxury hotel pool side with palms at night, skin reflects hotel in the desert surrounded by dark rugged terrain and towering volcanic peaks. She wears high-fashion clothing, contrasting with the dramatic landscape. Her hair flows gently in the wind as she gazes into the distance, under a moody sky with soft light breaking through the clouds. The scene blends natural beauty with modern glamour, highlighting the model against the striking volcanic background." \
--benchmark='True' \
--LoRA='True'
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

## comms with DB

```bash
source ./env
export FLASK_ENV='alpha
history|grep psql
psql ${DB_NAME}
select * from process_queue where id = 1

```
