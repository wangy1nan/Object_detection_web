from flask import Flask
from flask import request
import json
import config
from common.tools import *
import queue
import threading
import asyncio

put_url_que = queue.Queue()


app = Flask(__name__)

@app.route('/photo/putImgFileList/',methods=['GET', 'POST'])
def index():
    global put_url_que
    if request.json is None:
        return json.dumps({'code':1,'msg':"None_value"})
    # print(request.json)
    # print(request.method)
    # k = str(request.json)
    put_url_que.put(request.json)
    print('from post' + put_url_que.get())
    return json.dumps({'code':0,'msg':"succeed"})
    # return '<h1>Hello World!</h1>'
def post_scheduling():
    global t1
    t1 = threading.Thread(target=post_schealing)
    t1.start()
def down_load_img():
    down_load = asyncio.new_event_loop()
    srv_thread = threading.Thread(target=start_loop_forever,args=(down_load,))
    srv_thread.start()
    asyncio.run_coroutine_threadsafe(init(down_load), down_load)
    print('给系统2分钟下载图片的时间.....')
    time.sleep(5)


    # down_load = asyncio.new_event_loop()
    # srv_thread = threading.Thread(target=start_loop_forever,args=(down_load,))
    # srv_thread.start()
    # asyncio.run_coroutine_threadsafe(init(down_load,put_url_que), down_load)
    # print('给系统2分钟下载图片的时间.....')
    # time.sleep(5)
def detection_img():
    pass
def reback_post():
    pass
def main():
   post_scheduling()
   down_load_img()
   print('ssssss')

   detection_img()
   reback_post()
if __name__ == '__main__':
    main()
    app.run(debug=False,host=config.base_server_ip_host,port=config.base_server_ip_port)