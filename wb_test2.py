import random
import asyncio
from aiohttp import ClientSession
import json
import aiohttp



num = 0
async def fetch(url):
    global num
    async with ClientSession() as session:
        # data = {"id":int(num),'source':'tets2'}
        # num+=1
        k = json.dumps({'node': 7, 'dataList': [{'url': 'http://192.168.1.100:8080/static/20180713140001000.jpg'},{'url': 'http://192.168.1.100:8080/static/80AF38BF_20180824134041.jpg'}]})
        async with session.post(url, json=k) as response:
            # print(response.text())
            # delay = response.headers.get("DELAY")
            # date = response.headers.get("DATE")
            # print("{}:{} with delay {}".format(date, response.url, delay))
            # response.encoding="utf-8"
            string = await response.read()
            print(string)
            return string
            # return {"id":"hh",'source':'test3'}

async def bound_fetch(sem, url):
    # getter function with semaphore
    async with sem:
        await fetch(url)
async def run(r):
    url = "http://192.168.1.100:5050/"
    tasks = []
    # create instance of Semaphore
    sem = asyncio.Semaphore(1000)
    for i in range(r):
    # pass Semaphore to every GET request
        task = asyncio.ensure_future(bound_fetch(sem, url))
        tasks.append(task)
        responses = asyncio.gather(*tasks)
        await responses

number = 2
loop = asyncio.get_event_loop()
future = asyncio.ensure_future(run(number))
loop.run_until_complete(future)
# loop.run_forever()




# import requests
# import json

# import aiohttp
# import asyncio
#
# conn = aiohttp.TCPConnector(limit=1000)
# session = aiohttp.ClientSession(connector=conn)
#
# async def send_img_post(url):
#     async with session as sess:
#         body = {"callback_url": "http://192.168.1.101:5000/static"}
#         async with sess.post(url,data=body) as response:
#             res = await response.read()
#             print(res)
#
# loop = asyncio.get_event_loop()
# task = [send_img_post("http://192.168.1.106:9000/") for i in range(10)]
# loop.run_until_complete(asyncio.wait(task))
#
