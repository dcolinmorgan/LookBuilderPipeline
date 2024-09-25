from setuptools import setup, find_packages

setup(
    name="LookBuilderPipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your common dependencies here
        "flask",
        "Werkzeug",
        "Pillow",
        "numpy",
        "torch==1.9.0",
        "transformers==4.10.0",
        "huggingface-hub",
        "controlnet_aux==0.0.9",
        "opencv-python",

    ],
    extras_require={
        "flux": ["diffusers==0.30.2"],
        "sdxl": ["diffusers==0.27.2"],
    },
)
