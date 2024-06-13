#Author: Abdella Osman

from prediction import predict_image
from api_request import get_food_info

# Predict the class of the image
image_path = "sample_pics/03.jpg"
predicted_class = predict_image(image_path)

# Get food information based on the predicted class
get_food_info(predicted_class)
