#Author: Abdella Osman

from prediction import predict_image
from api_request import get_food_info
from PIL import Image

# Predict the class of the image
image_path = 'sample_pics/01.jpg'

image = Image.open(image_path)
predicted_class = predict_image(image)

# Get food information based on the predicted class
get_food_info(predicted_class)
