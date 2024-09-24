from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import base64
from PIL import Image

# Import your image model classes
from image_model_sdxl import ImageModelSDXL
from image_model_sd3 import ImageModelSD3
from image_model_fl2 import ImageModelFlux

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate_image', methods=['POST'])
def generate_image_api():
    if 'original_image' not in request.files or 'pose_image' not in request.files or 'mask_image' not in request.files:
        return jsonify({'error': 'Missing required image files'}), 400

    model_type = request.form.get('model', 'sdxl').lower()
    prompt = request.form.get('prompt', '')

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
        if model_type == 'sdxl':
            model = ImageModelSDXL(original_path, pose_path, mask_path, prompt)
        elif model_type == 'sd3':
            model = ImageModelSD3(original_path, pose_path, mask_path, prompt)
        elif model_type == 'flux':
            model = ImageModelFlux(original_path, pose_path, mask_path, prompt)
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
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded files
        for path in [original_path, pose_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
