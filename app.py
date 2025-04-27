import os
import re
import cv2
import numpy as np
import easyocr
import logging
from flask import Flask, request, render_template, jsonify
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename
from ChatGPT import get_llama_response, transcribe_audio, generate_audio_with_elevenlabs

# Configure Flask App
app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Logging
logging.basicConfig(level=logging.INFO)

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="i0ugmlchLJbER91K6IjS"
)

# EasyOCR Reader
reader = easyocr.Reader(['en'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    original_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    denoised_img = cv2.fastNlMeansDenoising(gray_img, None, h=30)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    enhanced_img = clahe.apply(denoised_img)
    _, thresh_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_img = cv2.filter2D(closed_img, -1, sharpen_kernel)
    processed_path = os.path.join(UPLOAD_FOLDER, "preprocessed.jpg")
    cv2.imwrite(processed_path, sharpened_img)
    return processed_path

def extract_text_from_image(image_path):
    return reader.readtext(image_path, detail=0)

def extract_name_and_aadhaar(text_list):
    aadhaar_number = None
    name = None
    combined_text = " ".join(text_list)
    match = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", combined_text)
    if match:
        aadhaar_number = match.group(1).replace(" ", "")
    candidates = [t for t in text_list if not re.search(r"\d", t) and len(t.split()) >= 2]
    if candidates:
        name = max(candidates, key=len)
    return name, aadhaar_number

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        logging.info(f"Image saved: {filename}")

        preprocessed_image_path = preprocess_image(image_path)

        try:
            result = CLIENT.infer(preprocessed_image_path, model_id="ourproject-ouuvb/1")
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return jsonify({"error": "Roboflow inference failed"}), 500

        detected_texts = []
        display_img = cv2.imread(preprocessed_image_path)
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        if "predictions" in result and result["predictions"]:
            for prediction in result["predictions"]:
                x = int(prediction["x"])
                y = int(prediction["y"])
                w = int(prediction["width"])
                h = int(prediction["height"])
                x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                x2, y2 = min(display_img.shape[1], x + w // 2), min(display_img.shape[0], y + h // 2)

                cropped_region = display_img_rgb[y1:y2, x1:x2].copy()
                cropped_path = os.path.join(UPLOAD_FOLDER, f"crop_{x1}_{y1}.jpg")
                cv2.imwrite(cropped_path, cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR))

                texts = extract_text_from_image(cropped_path)
                detected_texts.extend(texts)

                cv2.rectangle(display_img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        annotated_image_path = os.path.join(UPLOAD_FOLDER, "annotated.jpg")
        cv2.imwrite(annotated_image_path, cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2BGR))

        name, aadhaar_number = extract_name_and_aadhaar(detected_texts)

        return render_template("result.html",
                               name=name,
                               aadhaar_number=aadhaar_number,
                               annotated_image=annotated_image_path)

    return render_template("index.html")

@app.route("/submit_voice_data", methods=["POST"])
def submit_voice_data():
    voice_text = request.form.get("voice_text")
    return render_template("confirmation.html", voice_text=voice_text)

@app.route("/chat_llama", methods=["POST"])
def chat_llama():
    user_prompt = request.form.get("user_prompt")
    llama_response = get_llama_response(user_prompt) if user_prompt else None
    return jsonify({"llama_response": llama_response})

@app.route("/voice_to_llama", methods=["POST"])
def voice_to_llama():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_file.save(audio_path)

    transcript = transcribe_audio(audio_path)
    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    llama_reply = get_llama_response(transcript)
    voice_path = generate_audio_with_elevenlabs(llama_reply)

    return jsonify({
        "transcript": transcript,
        "llama_response": llama_reply,
        "audio_response_path": '/' + voice_path
    })

@app.route("/chat_llama_regional", methods=["POST"])
def chat_llama_regional():
    prompt = request.form.get("regional_prompt")
    if not prompt:
        return jsonify({"error": "No regional prompt provided"}), 400

    llama_reply = get_llama_response(prompt)
    audio_path = generate_audio_with_elevenlabs(text=llama_reply, output_path="static/llama_regional.mp3")

    return jsonify({
        "llama_response": llama_reply,
        "audio_url": '/' + audio_path
    })

if __name__ == "__main__":
    app.run(debug=True)
