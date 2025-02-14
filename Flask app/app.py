from flask import Flask, render_template, request
import os
from deeplearning import OCR

# webserever gateway interface

app = Flask(__name__)

# Get the current working directory (where the script is running)  
BASE_PATH = os.getcwd()
# Define the path where uploaded files will be saved (inside 'static/upload/' folder)
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

@app.route('/', methods=['GET', 'POST'])  # Define a route for the root URL ("/"), supporting both GET and POST requests
def index():  # Define the function to handle requests to "/"
    if request.method == 'POST':  # Check if the request method is POST (meaning a file was uploaded)
        Upload_file = request.files['image_name']  # Get the uploaded file from the request (input field named 'image_name')
        filename = Upload_file.filename  # Extract the filename of the uploaded file
        path_save = os.path.join(UPLOAD_PATH, filename)  # Create the full path where the file will be saved
        Upload_file.save(path_save)  # Save the uploaded file to the specified path
        text = OCR(path_save, filename)  # Call the 'OCR' function from 'deeplearning.py' to perform OCR on the uploaded file
        return render_template('index.html', upload=True, upload_image=filename)  # Render the 'index.html' template after saving the file
    
    return render_template('index.html', upload=False)  # If the request is GET, render the 'index.html' template without handling file upload

if __name__ == '__main__':
    app.run(debug=True)