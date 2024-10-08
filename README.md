# LookBuilderPipeline

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```
## Model Setup (via `setup.py` and `setup.sh`)
install will run the following commands:

```bash
!git clone https://github.com/Modegen/flux-controlnet-inpaint.git external_deps/
!git clone https://github.com/Modegen/ControlNetPlus.git external_deps/
!git clone https://${GH_PAT}@github.com/Modegen/LookBuilderPipeline.git
```

- diffusers will download `controlnet-union-sdxl-1.0-promax` for sdxl pipeline. However, the user needs to `controlnet-union-sdxl-1.0` and rename the folder to `controlnet-union-sdxl-1.0-promax`

- critical to add paths to these repos to the sys.path in the image model classes via the `external_deps` folder made via the [setup.sh](./setup.sh) script:
```python

# sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
sys.path.insert(0,os.path.abspath('external_deps/flux-controlnet-inpaint/src'))
sys.path.insert(1,os.path.abspath('external_deps/ControlNetPlus'))


## Running the tests

```bash
python -m pytest
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

