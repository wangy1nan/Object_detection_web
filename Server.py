# coding = utf-8

from flask import Flask
from flask import request
import json

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    # print(request.json)
    # print(request.method)

    # k = json.dumps({'node':1,'dataList':[{'url':'https://192.168.1.106:8000/1/0.jpg','alarmType':['digger']},
    #                                         {'url':'https://192.168.1.106:8000/1/1.jpg','alarmType':[ 'tower']}]})
    # s = json.loads(k)
    # print(s['dataList'])
    # return k
    return '<h1>Hello World!</h1>'
if __name__ == '__main__':
    app.run(debug=False,host='192.168.1.100',port=8000)