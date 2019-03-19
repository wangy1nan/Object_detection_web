# encoding: utf-8
"""
@author: nnn11556
@software: PyCharm
@file: server_test.py
@time: 2018/11/14 20:14
"""
from aiohttp import web
import json
import asyncio
import aiohttp
import threading
import aiofiles

conn = aiohttp.TCPConnector(limit=2000)
Session = aiohttp.ClientSession(connector=conn)
# import config
# CKPT = config.CKPT_PATH
URL = 'http://192.168.1.102:6000/static/4.jpg'
num = 0
async def handle_img(request):
    global savepath
    global num
    string = await request.json()
    print(string)
    #to do string url
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1000)) as sess:
        async with sess.post(URL) as resp:
            img = await resp.read()
            imgpath = str(num)+'.jpg'
            num+=1
            async with aiofiles.open(os.path.join(savepath,imgpath),'wb') as fp:
                await fp.write(img)
    return web.Response(body=b'<h1>My Bolg</h1>', content_type='text/html')


async def send_getimg_url(url,callback_url):
    global sched_Timer
    async with Session as sess:
        while True:
            now = int(time.time())
            if now == sched_Timer:
                print('post')
                async with sess.post(url,json=callback_url) as resp:
                    res = await resp.text()
                    #to do res
                    print(res)
                print('end')
                sched_Timer += span

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', handle_img)
    app.router.add_route('POST', '/', handle_img)
    srv = await loop.create_server(app.make_handler(), '192.168.1.102', 5050)
    print('start')
    return srv

# task = [send_getimg_url('http://192.168.1.101:5000',{'cabak':1})]
# loop2 = asyncio.get_event_loop()
# loop2.run_until_complete(asyncio.wait(task))
#
# loop1 = asyncio.get_event_loop()
# loop1.run_until_complete(init(loop1))
# loop1.run_forever()

new_loop = asyncio.new_event_loop()
srv_thread = threading.Thread(target=start_loop,args=(new_loop,))
srv_thread.start()

asyncio.run_coroutine_threadsafe(init(new_loop), new_loop)


import os
if not os.path.exists(r'result'):
    os.mkdir(r'result')
savepath = os.path.join(os.getcwd(),'result')
import datetime

import time
span = 10
sched_Timer = int(time.time()) + span
loop = asyncio.get_event_loop()
send_task = loop.create_task(send_getimg_url('http://192.168.1.100:5000', {'cabak': 1}))
loop.run_forever()

# loop.call_soon_threadsafe(send_task)
# while True:
#     now = int(time.time())
#     if now == sched_Timer:
#         print(now)
#         loop = asyncio.get_event_loop()
#         # loop.run_forever()
#         send_task = loop.create_task(send_getimg_url('http://192.168.1.106:5000', {'cabak': 1}))
#         loop.run_until_complete(send_task)
#         loop.stop()
#         # print(asyncio.Task.all_tasks())
#         sched_Timer += span


