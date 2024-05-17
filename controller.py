from flask import Flask, render_template, request, jsonify
from PIL import Image


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image']
    image = Image.open(data)
    print(image)
    prediction = "aaa"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()