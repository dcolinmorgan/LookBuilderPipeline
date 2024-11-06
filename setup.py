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
        'glob2',
        'dwpose',
        "bitsandbytes",
        'huggingface-hub',
        'sentencepiece==0.2.0',
        'diffusers @ git+https://github.com/huggingface/diffusers.git',
        'torchao',
        'sd_embed @ git+https://github.com/xhinker/sd_embed.git',
        'tokenizers',
        'protobuf',
        'controlnet_aux',
        'opencv-python',
        # 'flash-attn',# --no-build-isolation',
    ],
)
