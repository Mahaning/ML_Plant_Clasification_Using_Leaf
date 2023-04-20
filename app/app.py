# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup,redirect
import numpy as np
import pandas as pd
from utils.leaf import leaf_dic
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

# Loading plant disease classification model

leaf_classes = ['Apple',
                   'Apple',
                   'Apple',
                   'Apple',
                   'Blueberry',
                   'Cherry',
                   'Cherry',
                   'Corn',
                   'Corn',
                   'Corn',
                   'Corn',
                   'Grape',
                   'Grape',
                   'Grape',
                   'Grape',
                   'Orange',
                   'Peach',
                   'Peac',
                   'Pepper,',
                   'Pepper,_',
                   'Potato',
                   'Potato',
                   'Potato',
                   'Raspberry',
                   'Soybean',
                   'Squash',
                   'Strawberry',
                   'Strawberry',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato',
                   'Tomato']

leaf_model_path = 'models/plant_model.pth'
leaf_model = ResNet9(3, len(leaf_classes))
leaf_model.load_state_dict(torch.load(
    leaf_model_path, map_location=torch.device('cpu')))
leaf_model.eval()

# =========================================================================================
def predict_image(img, model=leaf_model):
    """
    Transforms image to tensor and predicts plant label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = leaf_classes[preds[0].item()]
    # Retrieve the class label
    return prediction
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

@ app.route('/')
def home():
    title = 'leaf - Home'
    return render_template('index.html', title=title)

@ app.route('/about')
def about():
    title = 'leaf - about'
    return render_template('about.html', title=title)

@ app.route('/contact')
def contact():
    title = 'leaf - contact'
    return render_template('contact.html', title=title)


@app.route('/leaf-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'leaf - Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('leaf.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
          
            prediction = Markup(str(leaf_dic[prediction]))
            return render_template('leaf-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('leaf.html', title=title)


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'leaf - identified'

    return render_template('fertilizer.html', title=title)

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'leaf - Crop Recommendation'
    return render_template('crop.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
