import os
import sys
import pathlib
import io
import base64
from flask import Flask, send_file, make_response
from flask import request
from torchvision import transforms
import torch
from PIL import Image
import subprocess


app = Flask(__name__, static_folder="../frontend/build", static_url_path='')
current_file_dir = pathlib.Path(__file__).parent.absolute()
test_file_path = current_file_dir.parent.parent / 'SR' / 'test.py'


@app.route("/")
def index():
    return app.send_static_file('index.html')


@app.route('/sr', methods=['POST'])
def sr():
    image = request.files['image']
    weight = request.files['weight']
    image_path = current_file_dir / 'data' / image.filename
    weight_path = current_file_dir / 'data' / weight.filename
    output_path = current_file_dir / 'data' / 'output.png'
    image.save(image_path)
    weight.save(weight_path)
    try:
        output = subprocess.check_output(f"python {test_file_path} -i {image_path} -o {output_path} -w {weight_path}",
                                         shell=True)
    except Exception as e:
        print(e)
        return e
    os.remove(image_path)
    os.remove(weight_path)
    with open(output_path, "rb") as f:
        image_binary = f.read()
        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition',
                             'attachment', filename='output.png')
    os.remove(output_path)
    return response


@app.route("/*")
def wild_card():
    return send_file('build/index.html')


if __name__ == '__main__':
    app.run()
