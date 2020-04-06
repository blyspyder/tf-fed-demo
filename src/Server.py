import tensorflow as tf
from tqdm import tqdm
from Client import Clients
import xlwt
import numpy as np


def gaussian_noise(input,std,client_id,sheet,j):
    length = len(input)
    sum=0
    for i in range(length):
        source = input[i].copy()
        #noise= tf.random_normal(shape=tf.shape(input[i]),mean=0.0,stddev=std,dtype=tf.float32)
        noise = np.random.normal(loc=0.0,scale=std,size=input[i].shape)
        input[i] += noise
        dist = np.linalg.norm(source-input[i])
        sum+=dist
    average = sum/length
    sheet.write(j,client_id,average)
    return input

def buildClients(num):
    learning_rate = 0.0005#0.0002
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    num_classes = 10  # Cifar-10 total classes (0-9 digits)

    #返回一定数量的clients
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)

def run_global_test(client, global_vars, test_num, i, save=False,sheet=None):
    #测试输出acc和loss
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num,save)
    sheet.write(i,0,float(acc))
    sheet.write(i,1,float(loss))
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))

scales = [0.0005,0.05,0.2]
for scale in scales:
    CLIENT_NUMBER = 4 #客户端数量
    '''可尝试更高比例的客户端'''
    CLIENT_RATIO_PER_ROUND = 0.5 #每轮挑选的clients的比例
    epoch = 260

    #### CREATE CLIENT AND LOAD DATASET ####
    client = buildClients(CLIENT_NUMBER)

    workbook = xlwt.Workbook()
    sheet=workbook.add_sheet('0.0002')
    #### BEGIN TRAINING ####
    sheet2 = workbook.add_sheet('欧式距离')

    global_vars = client.get_client_vars()
    for ep in range(epoch):
        #收集client端的参数
        client_vars_sum = None

        # 随机挑选client训练
        random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)

        # tqdm显示进度条
        for client_id in tqdm(random_clients, ascii=True):
            #将sever端模型加载到tqdm上
            client.set_global_vars(global_vars)

            # 训练这个下表的client
            client.train_epoch(cid=client_id)

            # 获取当前client的变量值
            current_client_vars_norm = client.get_client_vars()

            #获得参数后如高斯白噪声
            current_client_vars=gaussian_noise(current_client_vars_norm,scale,client_id,sheet2,ep)

            # 叠加各层参数
            if client_vars_sum is None:
                client_vars_sum = current_client_vars
            else:
                for cv, ccv in zip(client_vars_sum, current_client_vars):
                    cv += ccv

        # obtain the avg vars as global vars
        global_vars = []
        for var in client_vars_sum:
            global_vars.append(var / len(random_clients))

        # 测试集进行测试
        run_global_test(client, global_vars, test_num=600,i=ep,sheet=sheet)#将结果写入到excel中
    workbook.save('加噪scale={}结果.xls'.format(scale))
    #### FINAL TEST ####
    #run_global_test(client, global_vars, test_num=10000)
