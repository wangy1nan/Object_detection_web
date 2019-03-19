import json
import queue
import config


# url_List =queue.Queue()
# for i  in range(2):
#     k = json.dumps({'node':1,'url0':'https://192.168.1.100:8000/1/0.jpg','alarmType1':'[digger]',
#                                      'url1': 'https://192.168.1.100:8000/1/1.jpg', 'alarmType2': '[tower]'})
#     s = json.loads(k)
#     url = 'url' + str(i)
#     url_List.put(s[url])
# while not url_List.empty():
#     print( url_List.get())
# # print(url_List[0])


# k = json.dumps({'node':1,'dataList':[{'url':'https://192.168.1.100:8000/1/0.jpg','alarmType1':['digger']},
#                                      {'url': 'https://192.168.1.100:8000/1/1.jpg', 'alarmType2': ['tower']},]})
# def handle_url(callback_url):
#     url_List = []
#     url = json.loads(callback_url)
#     url_node = url['node']
#     num =  len(url['dataList'])
#     for i in range(num):
#         url_img = url['dataList'][i]['url']
#         url_List.append(url_img)
#     return url_node,url_List
# if __name__ == '__main__':
#     s,k = handle_url(k)
#     for item in k:
#         print(item)

from flask import *
from flask import request
import json
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print(request.json)
    return json.dumps({'hello':'hello_world'})

if __name__ == '__main__':
    app.run(debug=False,host='192.168.1.101', port=8080)