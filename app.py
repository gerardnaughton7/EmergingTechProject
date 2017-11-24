from flask import Flask, request, redirect, url_for, flash, render_template
from scipy.misc import imsave , imread, imresize
import numpy as np


# create a instance of a flask app
app = Flask(__name__)


# renders index.html
@app.route('/')
def index():
    return render_template("index.html")


 
if __name__ == '__main__':
        app.run()