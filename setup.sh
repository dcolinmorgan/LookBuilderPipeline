#!/bin/bash
# setup.sh

# Create a directory for external dependencies
mkdir -p external_deps
cd external_deps

# # Clone flux-controlnet-inpaint repository
# if [ ! -d "flux-controlnet-inpaint" ]; then
#     git clone https://${GH_PAT}@github.com/Modegen/flux-controlnet-inpaint.git
# else
#     echo "flux-controlnet-inpaint already exists, updating..."
#     cd flux-controlnet-inpaint
#     git pull
#     cd ..
# fi


# Clone ControlNetPlus repository
if [ ! -d "ControlNetPlus" ]; then
    git clone https://${GH_PAT}@github.com/Modegen/ControlNetPlus.git
else
    echo "ControlNetPlus already exists, updating..."
    cd ControlNetPlus
    git pull
    cd ..
fi

pip install -r requirements.txt
