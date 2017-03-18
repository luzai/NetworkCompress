import  matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def gray2rgb(gray):
    rgb = np.zeros((32, 32, 3), dtype = "uint8")
    rgb[:, :, 0] = gray
    rgb[:, :, 1] = gray
    rgb[:, :, 2] = gray
    return rgb


def get_transfer_data(transfer_data_path):

    #transfer_data_path = '/home/gx/NetworkCompress/data/transfer_data/'
    transfer_data_list_path = transfer_data_path + 'trans_data_list.txt'

    X_transfer = []
    Y_transfer = []

    file = open(transfer_data_list_path, 'r')

    nb_transfer_samples = len(file.readlines())
    X_transfer = np.zeros((nb_transfer_samples, 32, 32, 3), dtype="uint8")
    Y_transfer = np.zeros((nb_transfer_samples,), dtype="uint8")

    cnt = 0

    for line in file.readlines():
        tmp = line.split(' ')
        image_path = transfer_data_path + tmp[0]
        label = tmp[1]
        x = mpimg.imread(image_path)
        if x.shape == (32, 32):
            x = gray2rgb(x)

        X_transfer[cnt, :, :, :] = x
        Y_transfer[cnt] = label
        cnt = cnt + 1

    X_transfer = X_transfer.transpose((0, 3, 1,2))    

    print 'X_transfer shape is: ',  X_transfer.shape
    print 'Y_transfer shape is: ',  Y_transfer.shape
    
    return X_transfer, Y_transfer

