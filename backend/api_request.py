# Author: Abdella Osman

import requests
import json

# Function to get food information from Edamam API
def get_food_info(food_item):
    # app ID and app key from Edamam
    app_id = '2ea14798'
    app_key = 'ad00d45203bbf44b5ee30c5bcd56178f'

    # Construct the URL for the API request
    url = f'https://api.edamam.com/api/food-database/v2/parser?ingr={food_item}&app_id={app_id}&app_key={app_key}'

    # Make the GET request to the API
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON content of the response
        data = response.json()

        # Extract specific information if available
        if 'parsed' in data and data['parsed']:
            food = data['parsed'][0]['food']
        elif 'hints' in data and data['hints']:
            # If the parsed list is empty, use the first hint
            food = data['hints'][0]['food']
        else:
            food = None

        return food
    
    else:
        # Print the error status code if the request failed
        print(f"Error: {response.status_code} - {response.text}")
