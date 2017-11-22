import os
from flask import Flask, request, redirect, url_for, flash, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template("index.html")
 
if __name__ == '__main__':
        app.run()