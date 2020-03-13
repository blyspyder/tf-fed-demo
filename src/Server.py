import tensorflow as tf
from tqdm import tqdm
from Client import Clients
import xlwt

def buildClients(num):
    learning_rate = 0.001#0.0002
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

CLIENT_NUMBER = 4 #客户端数量
'''可尝试更高比例的客户端'''
CLIENT_RATIO_PER_ROUND = 0.5 #每轮挑选的clients的比例
epoch = 260

#### CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER)

workbook = xlwt.Workbook()
sheet=workbook.add_sheet('0.0002')
#### BEGIN TRAINING ####

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
        current_client_vars = client.get_client_vars()

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
    run_global_test(client, global_vars, test_num=500,i=ep,sheet=sheet)#将结果写入到excel中
workbook.save('未加噪结果.xls')
#### FINAL TEST ####
run_global_test(client, global_vars, test_num=10000)
