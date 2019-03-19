import tensorflow as tf
import numpy as np
# from resnets_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

np.seterr(invalid='ignore')

# os.chdir('D:\郭民鹏文件\data_text')
#-----------------------------------------------------------模拟数据集产生----------------------------------------------
def get_and_mix_data(filename):
    # 产生信号源的信号数据   shape=[20,500]
    file = pd.read_csv(filename)
    data = file.values
    signal_source = np.zeros((data.shape[0], 20))
    for i in range(0, 20):
        random_trend_signal = np.random.normal(size=(data.shape[0], 1))
        # 加上随机趋势信号
        temp = data + random_trend_signal
        signal_source[:, i] = temp[:, 0]
    signal_source = np.transpose(signal_source)
    return signal_source

#----------------------------------------------------------------------------------------------------------------------
def Position_creat(sample_num):
    t = np.random.random(size=sample_num) * 2 * np.pi - np.pi
    x = np.cos(t) * 10
    y = np.sin(t) * 10
    i_set = np.arange(0, sample_num, 1)
    for i in i_set:
        len = np.sqrt(np.random.random())
        x[i] = x[i] * len
        y[i] = y[i] * len
    return x,y

def Sensor_and_signal_source_position():
    # 拓扑，定位传感器和信号源的位置  shape=[20,2]  shape=[50,2]
    x_sensor,y_sensor = Position_creat(50)
    Sensor_position = np.hstack((x_sensor.reshape(50, 1), y_sensor.reshape(50, 1)))
    x_signal,y_signal = Position_creat(20)
    Signal_source_position = np.hstack((x_signal.reshape(20, 1), y_signal.reshape(20, 1)))
    return Sensor_position,Signal_source_position

def Find_two_shortest_distance(Sensor_position,Signal_source_position):
    # 找到距离每个传感器最近的两个信号源，返回信号源的编号  shape=[20,2]
    sort_index = np.zeros((20,1))
    y = np.zeros((50,2))
    for i in range(0,50):
        for j in range(0,20):
            distance = ((Sensor_position[i,0]-Signal_source_position[j,0])**2 + (Sensor_position[i,1]-Signal_source_position[j,1])**2)**(0.5)
            sort_index[j,0] = distance
        # y_temp = sort_index.argsort()   # 排序，返回的是index的值，index的值就是信号源的编号
        y_temp = np.argsort(sort_index,axis=0)
        y[i,0] = y_temp[0] # y的第一列保存最近的信号源的编号
        y[i,1] = y_temp[1] # y的第二列保存第二近的信号源的编号
    return y

def Data_creat(Sensor_position,Signal_source_position,Signal):
    y = Find_two_shortest_distance(Sensor_position,Signal_source_position)
    X = np.zeros((1,Signal.shape[1]))
    # 返回的X是无漂值  shape=[50,500]
    a = np.zeros((20,50))
    for j in range(0,20):
        for i in range(0,50):
            distance = ((Signal_source_position[j,0]-Sensor_position[i,0])**2 + (Signal_source_position[j,1]-Sensor_position[i,1])**2)**(0.5)
            a_ji = (distance+1)**(-1.5)
            a[j,i] = a_ji
    for i in range(0,50):
        temp = 0
        add = np.zeros((1,Signal.shape[1]))
        for j in range(0,20):
            temp = a[j,i] * Signal[j,:]
            add = temp + add
        # add是第一项  shape=[1,t]
        j_1 = y[i,0]
        j_2 = y[i,1]
        data_1 = a[int(j_1),i] * Signal[int(j_1),:]
        data_2 = a[int(j_2),i] * Signal[int(j_2),:]
        data = np.array([data_1*data_2])
        # 第二项  shape=[1,t]
        for k in range(add.shape[1]):
            add[0,k] = (add[0,k])**(0.5)
        for l in range(data.shape[1]):
            data[0,l] = (data[0,l])**(0.5)
        x_it = np.array(add + data)
        if i==0:
            X = x_it
        else:
            X = np.vstack((X, x_it))
    return X
# --------------------------------------------------------------数据生成------------------------------------------------
# 定义隔断数据的生成，返回 地面真实无漂数据 和 包含噪声和漂移的模拟数据
def Data_Generation(N,Tp,TI,X):
    # XI代表总的输入数据大小
    v = np.random.normal(loc=0,scale=0.5,size=(N,Tp))
    beta = np.random.normal(loc=0,scale=0.2)
    v = v + beta
    # 模拟噪声，高斯白
    d = np.zeros((N,Tp))
    delta = np.zeros((N,Tp))
    beta = np.random.normal(loc=0,scale=0.2)
    for i in range(0,N):
        r = np.random.uniform()
        if r<=0.5:
            mu_i = np.random.normal(loc=0,scale=0.5)
            d[i,0] = mu_i + beta
            for t in range(1,Tp):
                delta[i,t] = np.random.normal(loc=0,scale=0.02)
                d[i,t] = d[i,t-1] + delta[i,t]
        else:
            d[i,:] = 0
    # d是模拟漂移量 v是噪声 X是无漂数据
    Tau = np.random.randint(0,TI-Tp) # [0,Ti-Tp)整数
    # 从0到TI-Tp中取值作为截取的数据段的开头
    Xp = X[:,Tau:Tau+Tp]
    # 从截取的开头得到的size为（N,Tp）的数据段
    Yp = Xp + d + v
    # 最终合成的包含了无漂数据和模拟漂的移和噪声的训练数据  shape=[50,20]
    return Xp,Yp
    # Xp是无漂数据，Yp是测量数据

#--------------------------------------------------------------投影层--------------------------------------------------
# 定义初始化权值函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 定义初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

def Projection_Layer(x,y,N,T,num_D,Loss_w,TRAINING):
    Projection_loss = 0
    # x = tf.placeholder(tf.float32, [N, T])
    # y = tf.placeholder(tf.float32, [N, T])
    x_data = tf.reshape(x,[-1,N,T,1])
    y_data = tf.reshape(y,[-1,N,T,1])

    projection_w = weight_variable([N,7,1,2*N]) # R=2N 卷积核个数为2N

    projection_conv = conv2d(x_data,projection_w)
    projection_conv = tf.layers.batch_normalization(projection_conv, axis=3, training=TRAINING)
    projection_conv = tf.nn.tanh(conv2d(x_data,projection_w))
    #投影层卷积 nonlinear activation:tanh()
    drift_w = weight_variable([N,7,1,2*N]) # R=2N 卷积核个数为40
    drift_conv = conv2d(y_data, drift_w)
    drift_conv = tf.layers.batch_normalization(drift_conv, axis=3, training=TRAINING)
    drift_conv = tf.nn.tanh(drift_conv)

    # for i in range(0,num_D):
    #     temp = (LA.norm(projection_conv - drift_conv,'fro'))**(2)
        # LA.norm 范数 'fro'F-范数
        # Projection_loss = Projection_loss + temp
        # 计算Loss
    Projection_loss = tf.norm(projection_conv-drift_conv,ord='euclidean')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(Projection_loss)
    return drift_conv,train_step,Projection_loss

    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # for i in range(100):
    #     _,mse_cost,W,drift_observation = sess.run([train_step,mse,projection_w,projection_conv],feed_dict={x:exclusive_drift_data,y:include_drift_data})
    #     print(mse_cost)
    # # print(W)
    # # print(measure.shape)
    # # print('取R=2N,即20*2=40\n')
    # # print(drift_observation.shape)


#-------------------------------------------------------重排Re-arrangement层--------------------------------------------
def Re_arrangement(position,N):
    s = []
    s.append(0)
    sensor = np.arange(1,N)
    # 0号传感器已经定义，所以是arrange(1,50)一共49个sensor
    i = 0
    iterm = 1
    while iterm < N:
        temp_dist = 9999
        temp_num = 0
        l_temp = 0
        for j in sensor:
            if sensor == []:
                break
            l_temp = l_temp + 1
            distance = (position[i, 0] - position[j, 0]) ** 2 + (position[i, 1] - position[j, 1]) ** 2
            if distance < temp_dist:
                temp_dist = distance
                temp_num = j
                l_del = l_temp
                # 记录最短距离的传感器的序号，并删除在sensor中的该序号
        s.append(temp_num)
        sensor = np.delete(sensor, l_del - 1)
        i = temp_num
        iterm = iterm + 1
    change = np.ones((1, N), dtype=int)
    s = (s + change).flatten()
    # print(s)
    # print('\n')
    M = np.zeros((N, N), dtype=int)
    for k in range(0, N):
        M[k, s[k] - 1] = 1
    M = np.mat(M, dtype=int)
    # print(M)
    # M是由s[k]计算所得的变换矩阵
    position_new = np.dot(M, position)
    # 重排结果
    M_inv = M.I.astype(int)
    # M的逆矩阵
    # position_trans = np.dot(M_inv, Result)
    # 反变换矩阵用于重排的复原，结果层。
    return M,M_inv,position_new
    # 返回重排的M，用于复原重排的M_inv，重排结果position_new，重排反变换的结果position_trans。

#----------------------------------------------------------恢复层Recovery-----------------------------------------------
# ResUnit之前先通过一个3X3不包含Relu的卷积层调整通道数
def conv3X3_layer(re_input):
    conv3X3_w = weight_variable([1, 1, 4, 16])
    conv3X3 = conv2d(conv1X1_new, conv3X3_w)
    return conv3X3

# ResUnit 恢复层中包含三个ResUnit单元，两个个identity_block，一个convolutional_block。
# identity_block中shortcut路径不作处理，直接加入输入
def identity_block(X_input, filters, stage, TRAINING):
    # 定义层的名字
    conv_name_base = 'res' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input
        # 第一层
        x = tf.layers.conv2d(X_input,filter1,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2a')
        # 和tf.nn.conv2d 功能相同，tf.layers.conv2d的参数更多，适合训练模型时用。
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2a',training=TRAINING)
        x = tf.nn.relu(x)

        # 第二层
        x = tf.layers.conv2d(x,filter2,kernel_size=(3,3),padding='same',name=conv_name_base+'2b')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2b',training=TRAINING)
        # training：是否在训练，测试时为False
        x = tf.nn.relu(x)

        # 第三层
        x = tf.layers.conv2d(x,filter3,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2c',training=TRAINING)
        x = tf.nn.relu(x)

    # 添加快捷路径的输入
    x_add_shortcut = tf.add(x,X_shortcut)
    return x_add_shortcut

# convolutional_block中shortcut路径含一个1X1的卷积层，调整通道数和Branch中的一样
def convolutional_block(X_input, filters, stage, TRAINING):
    # 定义层的名字
    conv_name_base = 'res' + str(stage) +  '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # 快捷层
        X_shortcut = tf.layers.conv2d(X_shortcut,filter3,(1,1),strides=(1,1),name=conv_name_base+'1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut,axis=3,name=bn_name_base+'1',training=TRAINING)
        X_shortcut = tf.nn.relu(X_shortcut)

        # 第一层
        x = tf.layers.conv2d(X_input,filter1,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2a',training=TRAINING)
        x = tf.nn.relu(x)

        # 第二层
        x = tf.layers.conv2d(x,filter2,kernel_size=(3,3),padding='same',name=conv_name_base+'2b')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2b',training=TRAINING)
        x = tf.nn.relu(x)

        # 第三层
        x = tf.layers.conv2d(x,filter3,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2c',training=TRAINING)
        x = tf.nn.relu(x)

        # 添加快捷输入
        X_add_shortcut = tf.add(X_shortcut,x)

    return X_add_shortcut

# ResUnit结束后通过一个1X1的卷积层，调整通道数
def conv1X1_layer(ResUnit_output,TRAINING):
    conv1X1_w = weight_variable([1, 1, 64, 1])
    conv1X1 = conv2d(ResUnit_output,conv1X1_w)
    conv1X1 = tf.layers.batch_normalization(conv1X1,axis=3,training=TRAINING)
    conv1X1 = tf.nn.relu(conv1X1)
    return conv1X1

#----------------------------------------------------------------主函数-------------------------------------------------
# def main():
#     # 设置训练参数
#     Norm = 0
#     N = 20
#     Tp = 20
#     num_D = 64
#     Loss_w = (2*num_D*N*Tp)**(-1)
#
#     # 获取信号源的值
#     Signal_source_data = get_and_mix_data('csvlist.csv')
#     # 记录信号源和传感器在范围内的位置
#     Sensor_position, Signal_source_position = Sensor_and_signal_source_position()
#     Two_short_source = Find_two_shortest_distance(Sensor_position,Signal_source_position)
#     # 得到输入数据X
#     X = Data_creat(Sensor_position,Signal_source_position,Two_short_source,Signal_source_data)
#     for i in range(0,num_D):
#         Non_drift_temp,Measure_temp = Data_Generation(N,Tp,X.shape[1],X)
#         if i==0:
#             Non_drift_Tp = Non_drift_temp
#             Measure_Tp = Measure_temp
#         else:
#             Non_drift_Tp = np.hstack((Non_drift_Tp,Non_drift_temp))
#             Measure_Tp = np.hstack((Measure_Tp,Measure_temp))
#     # Non_drift_Tp 和 Measure_Tp 就是将num_D个Tp数据拼接起来的无漂输入和测量输入
#     Only_drift_data = Measure_Tp - Non_drift_Tp
#     Measure_data = Measure_Tp
#     # 投影层
#     Projection_result,Projection_optimizer,Projection_loss = Projection_Layer(Only_drift_data,Measure_data,num_D,Loss_w,True)
#     # Reshape层
#     conv1X1_w = weight_variable([1,1,40,80])
#     conv1X1 = conv2d(Projection_result, conv1X1_w)
#     # 加入1X1卷积层得到(1,T,4N)
#     conv1X1_new = tf.reshape(conv1X1, [1, 20, 200, 4])
#     # Re-arrange层
#     M,M_inv,position_new,position_trans = Re_arrangement(Sensor_position)
#     # 返回重排的M，用于复原重排的M_inv，重排结果position_new，重排反变换的结果position_trans。
#     for i in range(len(conv1X1_new.shape[2])):
#         conv1X1_new[:,:,i] = np.dot(M,conv1X1_new[:,:,i])
#     # 3X3conv调整通道数
#     ResUnit_input = conv3X3_layer(conv1X1_new)
#     # 残差网络
#     filter = [16,16,64]
#     ResNet_1 = convolutional_block(ResUnit_input,filter,1,have_conv,True)
#     ResNet_2 = identity_block(ResNet_1,filter,2,no_conv,True)
#     ResNet_3 = identity_block(ResNet_2,filter,3,no_conv,True)
#     # 1X1conv调整通道数
#     Recovery_output = conv1X1_layer(ResNet_3)
#     # 反向Re-arrange层
#     Recovery_output = np.dot(M_inv,Recovery_output)
#     # 计算差值得到loss
#     for i in range(0,num_D):
#         temp = (LA.norm(Recovery_output[:,:,i] - X[:,:,i],'fro'))**2
#         Norm = Norm + temp
#     Recovery_loss = Loss_w * Norm
#     Train_step = tf.train.AdamOptimizer(1e-4).minimize(Recovery_loss)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for epoch in range(1000):
#             sess.run(Projection_optimizer,feed_dict={x:Only_drift_data,y:Measure_data})
#             sess.run(Train_step)
#             if epoch%200==0:
#                 print(epoch, sess.run(Recovery_output))

# if __name__=='__main__':
#     main()


Norm = 0
N = 50
Tp = 20
num_D = 64
Loss_w = (2*num_D*N*Tp)**(-1)

# 获取信号源的值
Signal_source_data = get_and_mix_data('csvlist.csv')
# 记录信号源和传感器在范围内的位置
Sensor_position, Signal_source_position = Sensor_and_signal_source_position()
# 得到输入数据X
X = Data_creat(Sensor_position,Signal_source_position,Signal_source_data)
Xp,Yp = Data_Generation(N,Tp,X.shape[1],X)
for i in range(0,num_D):
    Non_drift_temp,Measure_temp = Data_Generation(N,Tp,X.shape[1],X)
    if i==0:
        Non_drift_Tp = Non_drift_temp
        Measure_Tp = Measure_temp
    else:
        Non_drift_Tp = np.hstack((Non_drift_Tp,Non_drift_temp))
        Measure_Tp = np.hstack((Measure_Tp,Measure_temp))
# Non_drift_Tp 和 Measure_Tp 就是将num_D个Tp数据拼接起来的无漂输入和测量输入
Only_drift_data = Measure_Tp - Non_drift_Tp
Measure_data = Measure_Tp
T = Measure_data.shape[1]
# 投影层
padding = np.zeros((50, 3))
Only_drift_data = np.hstack((padding, Only_drift_data, padding))
# 进行列的填充处理，保证输出的维度达到要求
X_measure = Measure_data
Measure_data = np.hstack((padding, Measure_data, padding))
T_projection_layer = Measure_data.shape[1]
x = tf.placeholder(tf.float32,[N,T_projection_layer])
y = tf.placeholder(tf.float32,[N,T_projection_layer])
Projection_result,Projection_optimizer,Projection_loss = Projection_Layer(x,y,N,T_projection_layer,num_D,Loss_w,True)
# Reshape层
conv1X1_w = weight_variable([1,1,2*N,4*N])
conv1X1 = conv2d(Projection_result, conv1X1_w)
# 加入1X1卷积层得到(1,T,4N)
conv1X1_new = tf.reshape(conv1X1,[1,N,T,4])
# Re-arrange层
M,M_inv,position_new = Re_arrangement(Sensor_position,N)
# 返回重排的M，用于复原重排的M_inv，重排结果position_new。
reshape_conv = tf.reshape(conv1X1,[N,T*4])
M = M.astype('float32')
conv1X1_new = tf.reshape(tf.matmul(M,reshape_conv),[1,N,T,4])#调整类型 int变为float32

# 3X3conv调整通道数
ResUnit_input = conv3X3_layer(conv1X1_new)
# 残差网络
filter = [16,16,64]
ResNet_1 = convolutional_block(ResUnit_input,filter,1,True)
ResNet_2 = identity_block(ResNet_1,filter,2,True)
ResNet_3 = identity_block(ResNet_2,filter,3,True)
# 1X1conv调整通道数
Recovery_output = conv1X1_layer(ResNet_3,True)
# 反向Re-arrange层
reshape_recov = tf.reshape(Recovery_output,[N,T])
M_inv = M_inv.astype('float32')
Recovery_output = tf.matmul(M_inv,reshape_recov)#调整类型 int变为float32
#
# 计算差值得到loss
Norm = tf.norm(Recovery_output - X_measure,ord='euclidean')
Recovery_loss = Loss_w * Norm
Train_step = tf.train.AdamOptimizer(1e-4).minimize(Recovery_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        result_1 = sess.run(Projection_optimizer,feed_dict={x:Only_drift_data,y:Measure_data})
        # result_2 = sess.run(Train_step,feed_dict={x:Only_drift_data,y:Measure_data})
        if epoch%200==0:
            print(epoch, sess.run(Recovery_output,feed_dict={x:Only_drift_data,y:Measure_data}))
            print(result_1)
            # print(result_2)
