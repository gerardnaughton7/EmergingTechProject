# Gerard Naughton G00209309 app.py EmergingTechProject
# Code adapted from https://github.com/sleepokay/mnist-flask-app
# Comments were derived/learned and quoted from keras website: https://keras.io/
# Imports required for program
from flask import Flask, render_template, request
from scipy.misc import imsave , imread, imresize
import numpy as np
import keras.models
import re
import base64

# create a instance of a flask app
app = Flask(__name__)

# renders index.html
@app.route('/')
def index():
    return render_template("index.html")

# Predict function
@app.route('/predict/', methods=['GET','POST'])
def predict():

    # get data from drawing canvas and save as image
    parseImg(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # load model to predict number
    model = keras.models.load_model("model/mnist_model.h5")

    # Use predict function and pass image x through it to get answer
    out = model.predict(x)
    print(out)
    # change out to number string
    response = np.array_str(np.argmax(out, axis=1))
    print(response)
    return response

# Parsing Image function
def parseImg(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug = True)