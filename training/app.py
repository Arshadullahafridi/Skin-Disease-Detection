from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='../templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "skin_disease_model.h5")

model = load_model(model_path)

classes = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot',
    'FU-nail-fungus', 'FU-ringworm', 'normal',
    'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # 🔥 Convert to PIL Image
        img = Image.open(BytesIO(file.read()))
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)

        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx] * 100)
        predicted_class = classes[class_idx]

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)