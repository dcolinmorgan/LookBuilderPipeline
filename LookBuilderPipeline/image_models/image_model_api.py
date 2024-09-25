from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os,sys
import io
import base64
from PIL import Image
# sys.path += ['external_deps/ControlNetPlus','external_deps/flux-controlnet-inpaint/src']
sys.path.insert(0,os.path.abspath('external_deps/flux-controlnet-inpaint/src'))
sys.path.insert(1,os.path.abspath('external_deps/ControlNetPlus'))
# Import your image model classes
# from image_model_sdxl import ImageModelSDXL
# from image_model_sd3 import ImageModelSD3
# from image_model_fl2 import ImageModelFlux
import sys

# if 'flux' in sys.modules:
#     from .flux_model import ImageModelFlux
# elif 'sdxl' in sys.modules:
#     from .sdxl_model import ImageModelSDXL
# else:
#     raise ImportError("Neither 'flux' nor 'sdxl' extras are installed")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate_image', methods=['POST'])
def generate_image_api():
    print(request.form)  # Debug: print the entire form data

    if 'original_image' not in request.files or 'pose_image' not in request.files or 'mask_image' not in request.files:
        return jsonify({'error': 'Missing required image files'}), 400

    model_type = request.form.get('model', '').lower()
    in_prompt = request.form.get('prompt')

    original_image = request.files['original_image']
    pose_image = request.files['pose_image']
    mask_image = request.files['mask_image']

    if not all(allowed_file(f.filename) for f in [original_image, pose_image, mask_image]):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save uploaded files
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(original_image.filename))
    pose_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pose_image.filename))
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(mask_image.filename))

    original_image.save(original_path)
    pose_image.save(pose_path)
    mask_image.save(mask_path)

    try:
        # Select the appropriate model
        if model_type == 'sdxl':# and 'sdxl' in sys.modules:
            from image_model_sdxl import ImageModelSDXL
            model = ImageModelSDXL(original_path, pose_path, mask_path, in_prompt)
        # elif model_type == 'sd3' and 'sd3' in sys.modules:
            # model = ImageModelSD3(original_path, pose_path, mask_path, prompt)
        elif model_type == 'flux': #and 'flux' in sys.modules:
            from image_model_fl2 import ImageModelFlux
            model = ImageModelFlux(original_path, pose_path, mask_path, in_prompt)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        # Generate the image
        generated_image = model.generate_image()

        # Convert the PIL Image to base64
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'image': encoded_image}), 200

    except Exception as e:
        return jsonify({'error2': str(e)}), 500

    finally:
        # Clean up uploaded files
        for path in [original_path, pose_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)

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
