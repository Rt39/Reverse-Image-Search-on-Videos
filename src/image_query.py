# Image query backend

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os

import torch
import torch.nn as nn
from PIL import Image
import psycopg2
import numpy as np


app = Flask(__name__)

# global variables for model
model_path = os.path.join("..", "data", "models", "encoder.pth")
encoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global variables for database
conn = None
cur = None

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')   
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')  
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same') 
        self.fc1 = nn.Linear(64 * 10 * 10, 64 * 5 * 5)
        self.fc2 = nn.Linear(64 * 5 * 5, 224)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))    # (3, 84, 84) -> (16, 84, 84)
        x = self.pool(x)                # (16, 84, 84) -> (16, 42, 42)
        x = self.relu(self.conv2(x))    # (16, 42, 42) -> (32, 42, 42)
        x = self.pool(x)                # (32, 21, 21) -> (32, 21, 21)
        x = self.relu(self.conv3(x))    # (32, 21, 21) -> (64, 21, 21)
        x = self.pool(x)                # (64, 21, 21) -> (64, 10, 10)
        x = x.reshape(-1, 64 * 10 * 10)   # (64, 10, 10) -> (64 * 10 * 10)
        x = self.relu(self.fc1(x))      # (64 * 10 * 10) -> (64 * 5 * 5)
        x = self.relu(self.fc2(x))       # (64 * 5 * 5) -> (224, )
        return x


@app.route('/')
def home():
    return render_template('index.html')

# Ensure there's a folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATA_IMG_FOLDER = os.path.join('..', 'data', "obj_imgs")
app.config['DATA_IMG_FOLDER'] = DATA_IMG_FOLDER

@app.route('/img/<path:img_path>', methods=['GET'])
def get_img(img_path):
    return send_from_directory(app.config['DATA_IMG_FOLDER'], img_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Process the file and query the database
        return process_and_query_image(filepath)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def process_and_query_image(image_path: str) -> jsonify:
    # Load and preprocess the image
    image = image_preprocessing(image_path)

    # Get the image embedding
    image = image.to(device)
    image_embedding = encoder(image)
    image_embedding = image_embedding.detach().cpu().numpy()

    # Query the database
    results = query_database(image_embedding)
    parsed_results = parse_sql_results(results)

    # Delete the uploaded image
    os.remove(image_path)

    return jsonify(parsed_results)


def load_encoder():
    global encoder
    encoder = Encoder()
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    encoder.to(device)

def connect_to_database():
    global conn
    conn = psycopg2.connect(
        dbname="postgres",
        password="postgres",
        user="postgres",
        host="db",
    )
    global cur
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

def query_database(image_embedding: np.ndarray) -> list:
    # Search for the most similar image in the database by cosine similarity
    search_query = """
    SELECT vidid, frameNum, timestamp, detectedObjClass, img_name,
    1 - (embedding <=> CAST(%s AS VECTOR(224))) AS cosine_similarity
    FROM video_embedded
    ORDER BY cosine_similarity DESC
    LIMIT 10;
    """
    global cur
    cur.execute(search_query, (image_embedding.flatten().tolist(),))
    results = cur.fetchall()
    return results

def image_preprocessing(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image = image.resize((84, 84))

    # Remove alpha channel if present
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    return image

def parse_sql_results(results: list) -> list:
    # map column names to index
    column2index = {"vidid": 0, "frameNum": 1, "timestamp": 2, "detectedObjClass": 3, "img_name": 4, "cosine_similarity": 5}
    parsed_results = []
    for row in results:
        parsed_row = {}
        for column, index in column2index.items():
            parsed_row[column] = row[index]
        parsed_results.append(parsed_row)
    return parsed_results

def disconnect_from_database():
    global cur
    cur.close()
    global conn
    conn.close()

def main():
    load_encoder()
    connect_to_database()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
    disconnect_from_database()

if __name__ == "__main__":
    main()