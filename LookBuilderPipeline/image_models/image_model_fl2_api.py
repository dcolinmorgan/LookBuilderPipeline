from flask import Flask, request, jsonify, send_file
from image_model_fl2 import ImageModelFlux
from base_image_model import generate_image_extras
import io
import base64
from PIL import Image

app = Flask(__name__)

@app.route('/generate_image_kids', methods=['POST'])
def generate_image_kids_api():
    # Get input data from the request
    data = request.json
    image = data.get('image')
    pose,mask,cannygenerate_image_extras(image)
    # pose = data.get('pose')
    # mask = data.get('mask')
    # canny = data.get('canny')
    prompt = data.get('prompt')
    
    # Create an instance of ImageModelFlux
    image_model = ImageModelFlux(image, pose, mask, canny, prompt)
    
    # Generate the image
    generated_image = image_model.generate_image_kids()
    
    # Convert the PIL Image to bytes
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Encode the image as base64
    encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    # Return the encoded image in the response
    return jsonify({'image': encoded_image})

if __name__ == '__main__':
    app.run(port=5005)
