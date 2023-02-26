import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont

from models.experimental import attempt_load
from utils.general import non_max_suppression

app = Flask(__name__)

WEIGHTS = "best.pt"
DEVICE = "cpu"
IMAGE_SIZE = 640

CLASSES = [
"A","B","Boss","C","D","E","F","Father","G","Good","H","I","J","L","M","Me","Mine","Mother","N","O","Onion","P","Q","Quiet","R","Responsible","S","Serious","T","Think","This","U","V","W","Wait","Water","X","Y","You","Z"
]

model = attempt_load(WEIGHTS, map_location=DEVICE)

def infer(image, image_size=640):
    image = np.asarray(image)
    
    # Resize image to the inference size
    ori_h, ori_w = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))
    
    # Transform image from numpy to torch format
    image_pt = torch.from_numpy(image).permute(2, 0, 1).to(DEVICE)
    image_pt = image_pt.float() / 255.0
    
    # Infer
    with torch.no_grad():
        pred = model(image_pt[None], augment=False)[0]
    
    # NMS
    pred = non_max_suppression(pred)[0].cpu().numpy()
    
    # Resize boxes to the original image size
    pred[:, [0, 2]] *= ori_w / image_size
    pred[:, [1, 3]] *= ori_h / image_size
    
    return pred


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image = Image.open(image)
    pred = infer(image)

    for x1, y1, x2, y2, conf, class_id in pred:
        label = CLASSES[int(class_id)]
        confidence = str(round(conf,4))
        break

    return jsonify({'class': label, 'confidence': confidence})


if __name__ == '__main__':
    app.run()