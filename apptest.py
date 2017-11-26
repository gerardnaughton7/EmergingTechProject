# Gerard Naughton G00209309 train.py EmergingTechProject

# Imports required for program
from __future__ import print_function
import keras
from keras.datasets import mnist
from scipy.misc import imsave, imread, imresize
import numpy as np

# read parsed image back in 8-bit, black and white mode (L)
x = imread('static/train-46-8.png', mode = 'L')
x = np.invert(x)
#x = imresize(x,(28,28))

x = x.reshape(1,28,28,1)
# Load the model again with: 
model = keras.models.load_model("model/mnist_model.h5")

out = model.predict(x)
print(out)
print(np.argmax(out, axis=1))
response = np.array_str(np.argmax(out, axis=1))
print(response)