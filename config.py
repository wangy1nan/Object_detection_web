# 本程序的流程：
# 以下数据可根据具体IP进行调整
##################################################################################
# 1.调度中心IP信息
scheduling_ip = 'http://10.104.2.89:8080/photo1/getImgFileList/'
# 2.本地开启监听IP
img_download_url = 'http://10.104.2.89:5050/'
# 3.本地检测完成回传的url
scheduling_callback_ip = 'http://10.104.2.89:8080/photo1/getImgFileList'

scheduling_ip_host = '10.104.2.89'
scheduling_ip_port =  8080
#####################################################################################
database_congfig = 'mysql+pymysql://root:123456@127.0.0.1:3306/web_img1'
#####################################################################################
# 相关文件存储目录
download_path = r'/photo/putImgFileList/'
absolute_img_path ='F:\Move'
log_img_path = 'F:\Move\\'
#####################################################################################
# 相关时间延迟
# 1.调度中心请求失败延迟时间
post_scheduling_time_sleep = 10
# 2.由于系统故障向调度中心回传检测图片结果失败的延迟
callback_scheduling_fail_time = 10
# 3.由于未检测完成向调度中心回传检测图片结果失败的延迟
undetection_callback_fail_time = 10
############################################################################
# 目标检测相关模型文件名称
# 1.模型名称
MODEL_NAME = 'inference_graph'
# 2.训练模型的文件名
PATH_TO_CKPT = r'F:\Move\frozen_inference_graph.pb'
# 3.训练模型映射
PATH_TO_LABELS = r'F:\Move\labelmap.pbtxt'
# 4.需要检测的最大类数目
NUM_CLASSES = 6