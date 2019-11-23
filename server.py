import sys
import argparse
import flask
from flask_cors import CORS
from torchvision import transforms, utils
from torch.utils.data.sampler import SequentialSampler
from PIL import Image
import torch
import numpy as np
from data.dataset import *
from models.inception import *
from flask import (
    Flask, request, jsonify
)

app = Flask(__name__)
CORS(app)

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--port', type=int,default=9000,help="localhost port")
    args = parser.parse_args()
    return args

@app.route('/classify', methods=['POST', 'GET'])
def classify():
    print("New Request")
    files = request.files
    dataset = RuntimeDataset(files)
    indices = list(range(len(files)))
    sampler = SequentialSampler(indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), sampler=sampler)

    result = trained_model.runtime_api(loader)
    print(result)
    return jsonify(result.tolist())

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'no home page'


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    args = arg_parse()
    model_path = "models/trained.pt"
    model = InceptionV3()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    trained_model = Trained_Model(model)
    app.run(host='127.0.0.1', port=args.port, debug=True)

