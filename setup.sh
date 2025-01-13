#!/bin/bash
# setup.sh

# Create a directory for external dependencies
mkdir -p external_deps
cd external_deps

# Clone DWPose repository
if [ ! -d "DWPose" ]; then
    git clone https://github.com/Modegen/DWPose.git LookBuilderPipeline/LookBuilderPipeline/
else
    echo "DWPose already exists, updating..."
    cd DWPose
    git pull
    cd ..
fi

# # Clone ControlNetPlus repository
# if [ ! -d "ControlNetPlus" ]; then
#     git clone https://${GH_PAT}@github.com/Modegen/ControlNetPlus.git
# else
#     echo "ControlNetPlus already exists, updating..."
#     cd ControlNetPlus
#     git pull
#     cd ..
# fi

pip install -r requirements.txt
