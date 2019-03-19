# encoding: utf-8
"""
@author: nnn11556
@software: PyCharm
@file: server_test.py
@time: 2018/11/14 20:14
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask
import os
from aiohttp import web
import asyncio
import aiohttp
import threading
import aiofiles
import time
import Object_detection_image
import config
import requests
import queue
from flask_sqlalchemy import SQLAlchemy
import shutil
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
sleep = 600
# 1.进行网络通信的准备
conn = aiohttp.TCPConnector(limit=2000)
Session = aiohttp.ClientSession(connector=conn)

Image_file_acquisition_interface = str(config.scheduling_ip_host) + r':' + str(config.scheduling_ip_port)
ip = config.img_download_url
x = ip.split('/')[-2]
My_web_server_IP = x.split(':')[0]
My_web_server_host = x.split(':')[1]


# 2.目标检测的必要配置文件
MODEL_NAME = config.MODEL_NAME
PATH_TO_CKPT = config.PATH_TO_CKPT
PATH_TO_LABELS = config.PATH_TO_LABELS
NUM_CLASSES = config.NUM_CLASSES

# 获得当前目录
savepath = os.path.join(os.getcwd())
#


#异步函数：处理接收到的图片url，将图片存入文件夹‘result’中
async def handle_img(request):
    global savepath
    global num
    string = await request.json()
    # 将从数据中心拿到的请求进行处理，结果为节点、该结点下的所有图片url列表
    node,url_List = handle_url(string)
    print('对原url进行如下处理：')
    print('数据结点： ' + str(node))
    print('图片url集合（列表）： ' + str(url_List))
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2000)) as sess:
        for item in url_List:
            # 从每个图片的url中提取图片的图片名
            print(item)
            print('从url集合中提取url用于储存图片 : '+ str(item))
            img = item.split('/')
            img_name = img[-1]
            # 给每个图片重命名（数据结点_原图片名）
            img_name_test = str(node)+'_' + img_name
            async with sess.get(item) as resp:
                img = await resp.read()
                # 将接收到的图片url保存到数据库中（download_report）
                tabel1 = download_img_table(time=str(datetime.datetime.now()),origin_path=item,down_load_img=os.path.join(savepath + r'/result/'+ img_name_test))
                db.session.add_all([tabel1])
                db.session.commit()
                # 将接收到的图片以新名称的方式保存到‘result’文件夹内
                async with aiofiles.open(os.path.join(savepath + r'/result', img_name_test),'wb') as fp:
                    await fp.write(img)
    # 写入数据库、保存图片成功向数据中心回传成功信号
    return web.json_response({'code': 0,'msg':'succeed'})

# 异步函数：将从所得的JSON数据提取出数据节点和url列表
def handle_url(database_url):
    print(database_url)
    url_node = database_url["node"]
    url_List = database_url["dataList"]
    return url_node,url_List
def loacl_detection_img():
    global read_img_que
    while True:
        while True:
            from collections import namedtuple
            image_t = namedtuple('img_t', ['url','path', 'data'])
            # 从数据库中查找未检测的图片
            put_img = download_img_table.query.filter_by(state=0).all()
            # print(put_img)
            # 将未检测图片读入队列中，并将数据库中的状态设为已检测（state = 1）
            for item in range(len(put_img)):
                url = put_img[item].down_load_img
                image_name = url.split('/')[-1]
                image = plt.imread(url)
                image_Np = np.array(image)
                tmp = image_t(url,image_name, image_Np)
                read_img_que.put(tmp)
                put_img[item].state = 1
                db.session.add(put_img[item])
            break
        # 将上述读入队列的图片进行检测
        while not read_img_que.empty():
            # 调用检测函数，进行本地检测
            obj = Object_detection_image.web_Object_detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
                                                              savepath, read_img_que)
            index_List, data_url,New_List = obj.detecion_img()
            url_database = []
            # 将检测完成的照片存入数据库detect_img_report中
            for i in New_List:
                detec_table = detect_img(time=str(datetime.datetime.now()), detect_img_url=i[0],
                                         detect_img_state=i[1],
                                         detect_img_label=i[2])
                url_database.append(detec_table)
            db.session.add_all(url_database)
            # 将检测结果以要求的格式进行打包
            respones_url = post_url_database(index_List, data_url)
            # 将打包好的数据存入数据库callback_report中
            if respones_url != []:
                callback_url = callback_database(time=str(datetime.datetime.now()), callback=str(respones_url))
                db.session.add(callback_url)
        db.session.commit()
#异步函数：用于将检测过的图片的url，发送回调度中心
async def send_getimg_url(url):
    while True:
        # 从数据库callback_report中查找未发送的url
        callback_report = callback_database.query.filter_by(state=0).all()
        # 将查找到的url一条条发送
        for k in range(len(callback_report)):
            try:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1000)) as sess:
                    callback = callback_report[k].callback
                    async with sess.post(url,json=callback) as resp:
                        res = await resp.text()
                        print('发送成功调度中心给的反馈： ' + str(res))
                    # 发送成功后，将数据库中的状态改为：已发送（state = 1）
                    callback_report[k].state = 1
                    db.session.add(callback_report[k])
            except Exception as e:
                    log = Logger('callback.log',level='debug')
                    log.logger.error(e)
                    print('系统故障，发送失败！！！！！')
                    time.sleep(config.callback_scheduling_fail_time)
        db.session.commit()
        print('照片还未检测完毕，请等待.....')
        time.sleep(config.undetection_callback_fail_time)
def post_url_database(index_List,data_url):
    result_response = []
    for i in index_List:
        data_List = []
        for item in data_url:
            url = item['url']
            index = url.split('/')[-2]
            if index == i:
                data_List.append(item)
        response = {'node': i, 'dataList': data_List}
        result_response.append(response)
    return result_response
def start_loop_forever(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()



# 异步函数：在调度路由中添加方法
async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', handle_img)
    app.router.add_route('POST', '/', handle_img)
    srv = await loop.create_server(app.make_handler(), My_web_server_IP, int(My_web_server_host))
    return srv

# 在请求发起成功之后，将本地开启的监听端口发往调度中心
def post_schealing():
    # post_log_path = os.path.join(config.absolute_img_path,r'post.log')
    # if not os.path.exists(post_log_path):
    #     os.mkdir(post_log_path)
    My_listener_ip = config.img_download_url
    My_listener_ip_1 = {'callback_url':str(My_listener_ip)}
    while True:
        try:
            requests.post(config.scheduling_ip, json=My_listener_ip_1)
            print('与调度中心连接成功......')
        except Exception as e:
            log = Logger('post.log',level='debug')
            log.logger.error(e)
            print('连接错误，系统将2分钟后重试......')
        time.sleep(config.post_scheduling_time_sleep)

# 程序正式开始：
###################################################################################################
###################################################################################################
# 为向数据中心请求开启一个线程
def post_scheduling():
    t1 = threading.Thread(target=post_schealing)
    t1.start()
# 为下载图片开启一个线程，并在本地开启一个监听端口，用以接收来源于数据中心的图片
def down_load_img():
    down_load = asyncio.new_event_loop()
    srv_thread = threading.Thread(target=start_loop_forever, args=(down_load,))
    srv_thread.start()
    asyncio.run_coroutine_threadsafe(init(down_load), down_load)
    print('给系统2分钟下载图片的时间.....')
    time.sleep(5)
#  向调度中心回传开启一个线程
def post_img_database():
    loop = asyncio.get_event_loop()
    send_task = loop.create_task(send_getimg_url(config.scheduling_callback_ip))
    loop.run_until_complete(send_task)
def read_img_local_datbase():
    read_img_local = threading.Thread(target=put_img_local_database)
    read_img_local.start()
def put_img_local_database():
    from collections import namedtuple
    image_t = namedtuple('img_t', ['path', 'data'])
    put_img = download_img_table.query.filter_by(state = 0).all()
    for i in range(len(put_img)):
        global read_img_que
        url = put_img[i].down_load_img
        image_name = url.split('/')[-1]
        image = plt.imread(url)
        image_Np = np.array(image)
        tmp = image_t(image_name, image_Np)
        read_img_que.put(tmp)
        put_img[i].state = 1
        db.session.add(put_img[i])
        db.session.commit()
def put_img():
    imgdir = os.getcwd()
    imgdir = os.path.join(imgdir + r'\result')
    print(imgdir)
    imgpath_lis = os.listdir(imgdir)
    from collections import namedtuple
    image_t = namedtuple('img_t', ['path', 'data'])

    begin = time.time()
    while imgpath_lis:
        for it in imgpath_lis:
            # img_que = queue.Queue()
            image = plt.imread(os.path.join(imgdir, it))
            image_Np = np.array(image)
            tmp = image_t(it, image_Np)
            read_img_que.put(tmp)
        break
    print('load data %f s' % (time.time() - begin))
app = Flask(__name__)
def read_img():
    read_file = threading.Thread(target=put_img)
    read_file.start()



db = SQLAlchemy(app)
class download_img_table(db.Model):
    __tablename__ = 'download_report'
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(64), unique=True)
    origin_path = db.Column(db.String(128))
    down_load_img = db.Column(db.String(128))
    state = db.Column(db.Integer, nullable=True, default=0, index=True)
    def __repr__(self):
        return '<download_report %r>' % self.id
class detect_img(db.Model):
    __tablename__ = 'detect_report'
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(64))
    detect_img_url = db.Column(db.String(128),)
    detect_img_state =db.Column(db.String(64))
    detect_img_label = db.Column(db.String(64))
    def __repr__(self):
        return '<detect_report %r>' % self.id

class callback_database(db.Model):
    __tablename__ = 'callback_report'
    id = db.Column(db.Integer, primary_key=True,index=True)
    time = db.Column(db.String(64),index=True)
    callback = db.Column(db.Text)
    state = db.Column(db.Integer, nullable=True, default=0, index=True)

    def __repr__(self):
        return '<detection_report %r>' % self.id
# 为本地检测图片开启一个线程
def local_detect():
    detect = threading.Thread(target=loacl_detection_img)
    detect.start()
#  为每天定时清理数据开启一个线程
def time_clean():
    clean = threading.Thread(target=clean_today_all_data)
    clean.start()
def clean_today_all_data():
    while True:
        # 每天刚过凌晨开始清理文件（已检测成功的图片、数据库中的信息）
        now = datetime.datetime.now()
        tomorrow = now.day + 1
        while now.day == tomorrow and now.hour == 0 and now.minute == 0 and now.second == 0:
            delete_report = download_img_table.query.filter_by(state=1).all()
            local_path = config.absolute_img_path
            detect_success_path = os.path.join(local_path,r'/result')
            # 删除从数据中心下载的原图
            for i in range(len(delete_report)):
                img_path = delete_report[i].down_load_img
                img_name = img_path.split('/')[-1]
                img_name = os.path.join(detect_success_path,img_name)
                if os.path.exists(img_name):
                    os.remove(img_name)
                else:
                    print('no file')
            # 删除检测完成的图片
            detected_img_path = os.path.join(local_path, r'/photo/putImgFileList/')
            if os.path.exists(detected_img_path):
                shutil.rmtree(detected_img_path)
                print('Successful delete file in %s' % detected_img_path)
            else:
                print('no file in %s' % detected_img_path)
            # 清空数据库
            db.drop_all()
            break

def main():
    # 1.创建数据库
    app.config['SQLALCHEMY_DATABASE_URI'] = config.database_congfig
    app.config['SQLALCHEMY_COMMIT_ON_TEAMDOWN'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    # app.config['SQLALCHEMY_ECHO'] = True
    db.drop_all()
    db.create_all()

    # 2. 向数据中心发送请求
    post_scheduling()
    # 3. 开启监听器，一旦有数据传入即刻进行图片下载，下载内容保存在‘result’文件夹，下载信息保存在数据库download_report表中。
    down_load_img()
    # 4. 对数据库中下载的图片进行检测，检测结果保存在数据库detect_report表中，生成的url存在 callback_report表中
    local_detect()
    # 5. 从callback_report表中提取数据发送
    post_img_database()
    # 6. 每天凌晨清理前一天的数据（数据库，保存图片文件夹）
    time_clean()

if __name__ == '__main__':
    read_img_que =queue.Queue()
    start_time = time.time()
    result_path = config.absolute_img_path + config.download_path
    down_load_path = os.path.join(config.absolute_img_path,r'/result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(down_load_path):
         os.makedirs(down_load_path)
    main()



