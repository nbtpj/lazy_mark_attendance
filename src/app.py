import threading
import face_recognition

from flask import Flask, request, jsonify, send_from_directory, Response, render_template_string
import numpy as np
import io
import cv2
import json
import os
from datetime import datetime
import socket
import queue

app = Flask(__name__, static_folder='static')

# File paths
frame_queue = queue.Queue()

DATA_DIR = 'data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

face_db = {}
named = []

today = datetime.now().strftime('%Y_%m_%d')
if os.path.exists(f'attendance_{today}.txt'):
    with open(f'attendance_{today}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            named.append(line.split(',')[0])


def load_face_database():
    """
    Load all saved faces (jpg + txt) from DATA_DIR into memory.
    Returns dict { identity: image_ndarray }.
    """
    db = {}
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".jpg"):
            base = os.path.splitext(filename)[0]
            img_path = os.path.join(DATA_DIR, filename)
            txt_path = os.path.join(DATA_DIR, base + ".txt")

            if not os.path.exists(txt_path):
                continue  # skip if no identity file

            # Load identity
            with open(txt_path, "r", encoding="utf-8") as f:
                identity = f.read().strip()

            if not identity:
                continue

            # Load image as numpy array (BGR format)
            img = cv2.imread(img_path)
            if img is None:
                continue  # skip corrupted images
            # obama_image = face_recognition.face_encodings(face_recognition.load_image_file(img_path))[0]
            # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
            face_encs = face_recognition.face_encodings(face_recognition.load_image_file(img_path))
            if len(face_encs) > 0:
                db[identity] = face_recognition.face_encodings(face_recognition.load_image_file(img_path))[0]

    return db


def update_face_database(new_identity, new_img_path):
    """
    Update the in-memory database with a new entry (or replace existing).
    new_identity: str
    new_img_path: str path to the saved image
    """
    if not os.path.exists(new_img_path):
        raise FileNotFoundError(f"Image not found: {new_img_path}")

    img = cv2.imread(new_img_path)
    if img is None:
        raise ValueError("Failed to load image")

    face_db[new_identity] = face_recognition.face_encodings(face_recognition.load_image_file(new_img_path))[0]


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'pure_check.html')


@app.route('/upload_face_ta', methods=['POST'])
def upload_face_ta():
    if 'face_image' not in request.files:
        return jsonify({'error': 'No face_image part'}), 400
    if 'identity' not in request.form:
        return jsonify({'error': 'No identity provided'}), 400

    face_image = request.files['face_image']
    identity = request.form['identity'].strip()

    if identity == '':
        return jsonify({'error': 'Empty identity'}), 400

    # Print identity in server console
    print(f'Received identity: {identity}')

    # Create unique filename based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base_filename = f'{timestamp}'

    # Save image file
    image_path = os.path.join(DATA_DIR, f'{base_filename}.jpg')
    face_image.save(image_path)

    update_face_database(identity, image_path)

    # Save identity text file
    identity_path = os.path.join(DATA_DIR, f'{base_filename}.txt')
    with open(identity_path, 'w', encoding='utf-8') as f:
        f.write(identity)

    return jsonify({'message': 'Face image and identity saved successfully'}), 200


@app.route('/get_named', methods=['GET'])
def get_named():
    print("Sent list", list(set(named)))
    return jsonify(list(set(named)))


@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'face_image' not in request.files:
        return 'No file', 400
    file = request.files['face_image']
    img_array = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    frame_queue.put(img)
    return 'OK', 200


def display_loop():
    today = datetime.now().strftime('%Y_%m_%d')

    while True:
        img = frame_queue.get()
        if img is None:
            break
        enc = face_recognition.face_encodings(img)
        text = "Unknown"
        if len(enc) > 0 and len(face_db) > 0:
            enc = enc[0]
            known_face_encodings = [v for v in face_db.values()]
            known_face_names = [k for k in face_db.keys()]
            matches = face_recognition.compare_faces(known_face_encodings, enc)
            face_distances = face_recognition.face_distance(known_face_encodings, enc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                text = name
                now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                if name not in named:
                    print(f'Found {name} with distance {face_distances[best_match_index]}')
                    with open(f'attendance_{today}.txt', 'a+', encoding='utf-8') as f:
                        f.write(f'{name}, {now}, {face_distances[best_match_index]}\n')
                # del face_db[name]

        # 1. Resize to 256x256
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # 2. Put text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # green text
        thickness = 2
        org = (10, 25)  # bottom-left corner of text

        cv2.putText(resized, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Received Face", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_db = load_face_database()
    lan_ip = socket.gethostbyname(socket.gethostname())
    threading.Thread(target=lambda: app.run(host=lan_ip, port=8080, ssl_context=("certs/cert.pem", "certs/key.pem"),
                                            threaded=True, debug=False), daemon=True).start()
    display_loop()
