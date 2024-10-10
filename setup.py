from setuptools import setup, find_packages

# class CloneRepoCommand(Command):
#     """Custom command to clone a repository."""
#     description = 'Clone external dependencies'
#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         repo_url = 'https://github.com/Modegen/ControlNetPlus.git'
#         repo_url2 = 'https://github.com/Modegen/flux-controlnet-inpaint.git'
#         clone_dir = 'external_deps'
#         if not os.path.exists(clone_dir):
#             subprocess.check_call(['git', 'clone', repo_url, clone_dir])
#             subprocess.check_call(['git', 'clone', repo_url2, clone_dir])
#         else:
#             print(f'Repository already cloned in {clone_dir}')


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
    # extras_require={
    #     "flux": ["diffusers==0.30.2"],
    #     "sdxl": ["diffusers==0.27.2"],
    # },
    # cmdclass={
    #     'clone_repo': CloneRepoCommand,
    # },
)
