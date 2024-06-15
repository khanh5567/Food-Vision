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

        # Print the simplified output if food information is found
        if food:
            return food

            # print(f"Food Label: {food['label']}")
            # print(f"Category: {food.get('category', 'N/A')}")
            # print(f"Calories: {food['nutrients'].get('ENERC_KCAL', 'N/A')} kcal")
            # print(f"Protein: {food['nutrients'].get('PROCNT', 'N/A')} g")
            # print(f"Fat: {food['nutrients'].get('FAT', 'N/A')} g")
            # print(f"Carbs: {food['nutrients'].get('CHOCDF', 'N/A')} g")
            # print(f"Fiber: {food['nutrients'].get('FIBTG', 'N/A')} g")
            # print(f"Cholesterol: {food['nutrients'].get('CHOLE', 'N/A')} mg")
            # print(f"Sodium: {food['nutrients'].get('NA', 'N/A')} mg")
            # print(f"Potassium: {food['nutrients'].get('K', 'N/A')} mg")
            # print(f"Calcium: {food['nutrients'].get('CA', 'N/A')} mg")
            # print(f"Iron: {food['nutrients'].get('FE', 'N/A')} mg")
            # print(f"Vitamin A: {food['nutrients'].get('VITA_RAE', 'N/A')} µg")
            # print(f"Vitamin C: {food['nutrients'].get('VITC', 'N/A')} mg")
            # print(f"Sugars: {food['nutrients'].get('SUGAR', 'N/A')} g")
            # # Extract and print the ingredients
            # ingredients = food.get('foodContentsLabel', 'N/A')
            # print(f"Ingredients: {ingredients}")
        else:
            print("No food information found.")
    else:
        # Print the error status code if the request failed
        print(f"Error: {response.status_code} - {response.text}")
