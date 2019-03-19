from flask import request
from flask import Flask
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
from web import *
import queue
Image_file_acquisition_interface = config.scheduling_ip_host
My_web_server_IP = config.base_ip_host
My_web_server_host = config.base_ip_port


async def handle_img(request):
    global savepath
    global num
    string = await request.json()
    print('从数据集采集到的原url ： ' + str(string))
    #to do string url
    node,url_List = handle_url(string)
    print('对原url进行如下处理：')
    print('数据结点： ' + str(node))
    print('图片url集合（列表）： ' + str(url_List))
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2000)) as sess:
        for item in url_List:
            print('从url集合中提取url用于储存图片 : '+ str(item))
            img = item.split('/')
            img_name = img[-1]
            # x = img_name.split('.')[0]
            # async with sess.post(item + '/static/1.jpg') as resp:
            img_name_test = str(node)+'_' + img_name
            async with sess.get(item) as resp:
                img = await resp.read()
                # imgpath = str(num)+'.jpg'
                async with aiofiles.open(os.path.join(savepath + r'/result', img_name_test),'wb') as fp:
                    await fp.write(img)
            # num += 1
    # print(web.Response(body=b'<h1>My Bolg</h1>', content_type='text/html'))
    # return web.Response(body=b'<h1>My Bolg</h1>', content_type='text/html')
    # return json.dumps({'code': 0})
    return web.json_response({'code': 0,'msg':'succeed'})
# 异步函数：将从所得的JSON数据提取出数据节点和url列表

def handle_url(scheduling_url):
    # url = json.loads(scheduling_url)
    # print(url)
    node,dataList = scheduling_url['node'],scheduling_url['dataList']
    return node,dataList

def post_schealing():
    My_listener_ip = str(config.base_ip_host) + r':' + str(config.base_ip_port)

    My_listener_ip_1 = {'callback_url':r'http://' + str(My_listener_ip) + r'/photo/putImgFileList/'}
    while True:
        try:
            # print(My_listener_ip_1)
            requests.post(Image_file_acquisition_interface, json=My_listener_ip_1)

            print('与调度中心连接成功......')
        except Exception as e:
            print(e)
            print('连接错误，系统将2分钟后重试......')
        time.sleep(5)
def start_loop_forever(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', handle_img)
    app.router.add_route('POST', '/', handle_img)
    # handle_img(handle_que)
    srv = await loop.create_server(app.make_handler(), My_web_server_IP, My_web_server_host)
    print('start')
    return srv
    # # srv = await loop.run_forever
    # return srv
