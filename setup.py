from setuptools import setup, find_packages
import platform
import os

# Only run system commands if not on macOS
if platform.system() != 'Darwin':  # Darwin is macOS
    try:
        os.system('apt-get update && apt-get install -y libgl1')
        os.system('git submodule add https://github.com/Modegen/DWPose.git LookBuilderPipeline/DWPose/')
        os.system('git submodule init && git submodule update')
    except Exception as e:
        print(f"Warning: System commands failed: {e}")

setup(
    name='LookBuilderPipeline',
    version='0.1',
    packages=find_packages(exclude=['debug_poses*', 'tests*']),
    install_requires=[
        'pillow',
        'transformers',
        'pytest',
        "matplotlib",
        'torch',
        'mediapipe',
        'accelerate',
        'numpy',
        'python-dotenv',
        'compel',
        'glob2',
        'bitsandbytes',
        'dwpose @git+https://github.com/Modegen/DWPose.git',
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
    ],
)
