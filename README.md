# LookBuilderPipeline

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```
## Model Setup
install will run the following commands:

```bash
git clone https://${GH_PAT}@github.com/Modegen/flux-controlnet-inpaint.git
git clone https://${GH_PAT}@github.com/Modegen/ControlNetPlus.git
```

- diffusers will download `controlnet-union-sdxl-1.0-promax` for sdxl pipeline. However, the user needs to `controlnet-union-sdxl-1.0` and rename the folder to `controlnet-union-sdxl-1.0-promax`

- critical to add paths to these repos to the sys.path in the image model classes via the `external_deps` folder made via the [setup.sh](./setup.sh) script:
```python
sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
```

## Running the tests

```bash
python -m pytest
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

