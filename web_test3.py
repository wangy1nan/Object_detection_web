# coding = utf-8

from flask import Flask
from flask import request
import json
import config
import os
from aiohttp import web
import asyncio
import aiohttp
import threading
import aiofiles
import json
import time
import Object_detection_image
import config
import requests


app = Flask(__name__)

@app.route('/photo/getImgFileList/',methods=['GET', 'POST'])
def index():
    print(request.json)
    print(request.method)
    return json.dumps({'code':0,'msg':"succeed"})
    # return '<h1>Hello World!</h1>'

if __name__ == '__main__':
    app.run(debug=False,host='192.168.1.101',port=8080)
