from setuptools import setup
import os

# Run shell command to install system dependencies
os.system('apt-get update && apt-get install -y libgl1')

setup(
    name='LookBuilderPipeline',
    version='0.1',
    install_requires=[
        'pillow',
        'transformers',
        'pytest',
        "matplotlib",
        'torch',
        'mediapipe',
        'accelerate',
        'numpy',
        'compel',
        'huggingface-hub',
        'sentencepiece==0.2.0',
        'diffusers @ git+https://github.com/huggingface/diffusers.git',
        'tokenizers',
        'protobuf',
        'controlnet_aux',
        'opencv-python',
    ],
)
