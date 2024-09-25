from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os,sys
import io
import base64
from PIL import Image
# sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
sys.path.insert(0,os.path.abspath('external_deps/flux-controlnet-inpaint/src'))
sys.path.insert(1,os.path.abspath('external_deps/ControlNetPlus'))

import sys

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate_images', methods=['POST'])
def generate_images_api():
    model_type = request.form.get('model', '').lower()
    in_prompt = request.form.get('prompt')
    
    # Define input folders
    original_folder = request.form.get('original_folder', 'input/original')
    pose_folder = request.form.get('pose_folder', 'input/pose')
    mask_folder = request.form.get('mask_folder', 'input/mask')

    try:
        # Select the appropriate model
        if model_type == 'sdxl':
            from image_model_sdxl import ImageModelSDXL as ImageModel
        elif model_type == 'flux':
            from image_model_fl2 import ImageModelFlux as ImageModel
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        model = ImageModel(original_folder, pose_folder, mask_folder, in_prompt)
        model.load_images()
        model.load_model()
        generated_images = model.run_model()
        model.save_image(generated_images, app.config['OUTPUT_FOLDER'])

        result = []
        for generated_image, original_filename in generated_images:
            output_filename = f"generated_{original_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Convert the PIL Image to base64
            buffered = io.BytesIO()
            generated_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            result.append({
                'original': original_filename,
                'generated': output_filename,
                'image': encoded_image
            })

        return jsonify({'generated_images': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)


# curl -X POST http://50.7.159.181:5005/generate_image\
#   -F "original_image=@'/Users/apple/WRK/modegen/input/orig/p09.jpg'" \
#   -F "pose_image=@'/Users/apple/WRK/modegen/input/pose/pose_p09.jpg'" \
#   -F "mask_image=@'/Users/apple/WRK/modegen/input/set/seg-p09.png'" \
#   -F "model=sdxl" \
#   -F "prompt='photo realistic female fashion model with blonde hair on paris street corner'" \
#   http://50.7.159.181:21272/generate_image


# curl -X POST \
#   -F "original_image=@../img/orig/p09.jpg" \
#   -F "pose_image=@../img/pose/pose-p09.jpg" \
#   -F "mask_image=@../img/seg/seg-p09.png" \
#   -F "model=flux" \
#   -F "prompt=photo realistic female fashion model with blonde hair on paris street corner" \
#   http://127.0.0.1:5005/generate_image
