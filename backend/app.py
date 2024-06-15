from flask import Flask, request, jsonify, send_from_directory #need import
from flask_cors import CORS # need import
from PIL import Image
from backend.prediction import predict_image
from backend.api_request import get_food_info

app = Flask(__name__, static_folder='frontend')
cors = CORS(app)

@app.route('/')
def serve_react_app():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    # If the file exists in the build directory, serve it
    if path and path.startswith('assets/'):
        return send_from_directory(app.static_folder, path)
    # If the requested path doesn't exist, serve index.html
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open the image file using Pillow directly from the file stream
        image = Image.open(file.stream)
        predicted_class = predict_image(image)

        # Get food information based on the predicted class
        #food_info_dict = get_food_info(predicted_class)

        response = jsonify({
            'predicted_class': predicted_class,
            #'nutrient': food_info_dict['nutrients'],
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

