

import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import config
import queue
from collections import namedtuple
from flask import Flask
import time




# 检测图片的载入准备
MODEL_NAME = config.MODEL_NAME
PATH_TO_CKPT = config.PATH_TO_CKPT
PATH_TO_LABELS = config.PATH_TO_LABELS
NUM_CLASSES = config.NUM_CLASSES
savepath = config.absolute_img_path
log_savepath = config.log_img_path


# sys.path.append("..")

# 为了便于调用，将图片检测定义为一个类
class web_Object_detection(object):


    # 创建日志写入文件
    now = datetime.datetime
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    # log_savepath = log_savepath  +str(log_name) + r'.txt'
    if not os.path.exists(log_savepath):
        log_txt = log_savepath
    # 初始化方法，将需要的参数载入模型中
    def __init__(self, PATH_TO_CKPT,PATH_TO_LABELS,NUM_CLASSES,savepath,read_img_que):
        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.PATH_TO_LABELS = PATH_TO_LABELS
        self.NUM_CLASSES = NUM_CLASSES
        self.savepath = savepath
        self.read_img_que= read_img_que




    # 正式开始检测图片并写入日志，将检测完成的图片存入各自节点的文件夹内
    def detecion_img(self):
        #载入映射
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        #
        def save_detect_image(img, path, savepath):
            image_np = img
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np.data,
                np.squeeze(image_np.box),
                np.squeeze(image_np.classes).astype(np.int32),
                np.squeeze(image_np.score),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            if scores[0][0] >= 0.8:
                pic_name = path.rstrip('.jpg')
                pic_name = pic_name + '_res.jpg'
                # print(os.path.join(savepath, pic_name))
                plt.imsave(os.path.join(savepath, pic_name), image_np.data, dpi=800)
            else:
                pic_name = path.rstrip('.jpg')
                pic_name = pic_name + '.jpg'
                plt.imsave(os.path.join(savepath, pic_name), image_np.data, dpi=800)
        # 检测程序开始
        print('begin')
        import time
        begin = time.time()
        # 将TensorFlow模型存到内存中.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        # 输出的张量为检测框、得分、类别
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 每个框表示检测到特定对象的图像的一部分
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # 每个分数代表每个对象的置信水平。.
        # 分数与标签一起显示在图片中.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # 检测到对象数
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # 载入模型结束
        print('load model end in {0}s'.format(time.time() - begin))


        index = 0
        # 得到保存于队列中的image_t元组对象，里面保存有图片路径和图片序列化后的数据
        read_img_que = self.read_img_que
        begin = time.time()
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(3)
        import threading
        # 建立新的元组：img_res中包含['data', 'box', 'score', 'classes', 'num']属性
        IMG_res = namedtuple('img_res', ['data', 'box', 'score', 'classes', 'num'])
        # 获取文件中图片的来源节点，用于分类存放
        Img_name_List = []
        # 新建队列用于保存原本数据
        detec_img_que = queue.Queue()
        # 将图片的数据节点有存到Img_name_List中
        while not read_img_que.empty():
            j = read_img_que.get()
            detec_img_que.put(j)
            index = j.path.split('_')[0]
            Img_name_List.append(index)
        # 将所有节点重复部分去掉得到index_List
        index_List = list(set(Img_name_List))
        New_List =[]
        data_url = []
        index_id = 0
        detect_database_list = []
        boxes_list = []
        # 将每个图片检测完成后结果保存到对应节点的文件夹内
        for i in index_List:
            if not os.path.exists(savepath + r'/photo/putImgFileList/' + str(i)):
                os.makedirs(savepath + r'/photo/putImgFileList/' + str(i))
            while not detec_img_que.empty():
                List = []
                it = detec_img_que.get()
                data_node = it.path.split('_')[0]
                data_List = []
                # 进行检测
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(it.data, axis=0)})
                tmp = IMG_res(it.data, boxes, scores, classes, num)
                name = it.path
                node = name.split('_')[0]
                # 拿到结果保存的路径
                saveimgpath = os.path.join(savepath + r'/photo/putImgFileList/' + str(node))
                # 检测结果可视化并保存到指定目录中
                pool.submit(save_detect_image, tmp, name, saveimgpath)
                # 判断结果图片的报警类型和报警结果（yes/no）
                if scores[0][0] >= 0.8:
                    alarm_state =  'yes'
                    alarm_boxes_list = boxes[0][0]
                    k = int(classes[0][0])
                    if k == 1:
                        alarmType = 'motocrane'
                    elif k == 2:
                        alarmType = 'towercrane'
                    elif k == 3:
                        alarmType = 'digger'
                    elif k == 4:
                        alarmType = 'pushdozer'
                    elif k == 5:
                        alarmType = 'smog'
                    else:
                        alarmType = 'other'
                else:
                    alarm_boxes_list = []
                    alarm_state = 'no'
                    alarmType = ''
                alarm_boxes_data = {'xmin': alarm_boxes_list[1], 'ymin': alarm_boxes_list[0],
                                    'xmax': alarm_boxes_list[3], 'ymax': alarm_boxes_list[2]}

                # 得到检测完成图片的url：new_url
                new_url = config.img_download_url + config.download_path + str(data_node) + r'/' + str(name)
                resp = {r'url': new_url,r'alarmTypeList': [{'type':alarmType,'bndbox':alarm_boxes_data}]}
                # 将部分信息保存到List列表中，用于后续数据库信息的添加
                List.append(new_url)
                List.append(alarm_state)
                List.append(alarmType)
                New_List.append(List)
                # boxes_list.append(alarm_boxes_data)
                # 将打包好的数据存入data_url，用于后续给调度中心回传信息
                data_url.append(resp)
                # 写相关日志
                read_log = r'[' + str(datetime.datetime.now()) +r']-[' + str(data_node) +r']-['\
                           + it.url +r']=>[' + new_url +r'][' + alarm_state + r'][' + alarmType +r']'
                log_name = time.localtime()
                log_name = time.strftime("%Y-%m-%d", log_name)
                write_log = open(log_savepath + log_name + r'.txt','a+',encoding='utf-8')
                write_log.writelines(read_log+"\n")
                write_log.close()

                # 一次检测完成
                index_id += 1
                print(index_id)
        print('finish ^_^ in {0}s'.format(time.time() - begin))
        return index_List,data_url,New_List




