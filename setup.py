from setuptools import setup, find_packages

setup(
    name="LookBuilderPipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your common dependencies here
        "flask==2.0.1",
        "Werkzeug==2.0.1",
        "Pillow==8.3.1",
        "numpy==1.21.0",
        "torch==1.9.0",
        "transformers==4.10.0",
        "huggingface-hub",
        "controlnet_aux==0.0.9",
        "opencv-python==4.5.3.56",

    ],
    extras_require={
        "flux": ["diffusers==0.30.2"],
        "sdxl": ["diffusers==0.27.2"],
    },
)
