import sys
import time
import numpy as np
import pandas as pd
import torch as tor
from train import NNModel



def prediction_nn(model_fp):
    data_test_fp_list = ['./data/prepro_merged_test_data_{}.npy'.format(i) for i in range(5)]

    model = NNModel()
    model.load_state_dict(tor.load(model_fp))
    model.eval()

    pred_array = np.array([])

    for test_data_fp in data_test_fp_list:
        data = tor.FloatTensor(np.load(test_data_fp).astype(np.float64))
        print('|Working on: {}'.format(test_data_fp))
        print('|Show test data samples')
        print(data[:5]) 

        pred = model(data)
        pred = pred.detach().numpy().reshape(-1)
        print(pred)
        pred_array = np.hstack((pred_array, pred))
    
    pred_array = pred_array.reshape(-1, 1)
    print('|Prediction size: {}'.format(pred_array.shape[0]))

    pd.DataFrame(pred_array).to_csv('./prediction_{}_{}.csv'.format(mode, int(time.time())), index_label='id', header=['target'])


def prediction_stat(model_fp):
    data_train = np.array(pd.read_csv('./data/train.csv'))
    data_test_all_fp = './data/test.csv'
    data_test_fp_list = ['./data/prepro_merged_test_data_{}.npy'.format(i) for i in range(5)]
    data_test_all = np.array(pd.read_csv(data_test_all_fp, index_col=0))
    output_array = np.empty((len(data_test_all)))

    # Build user dict
    user_dict = dict()

    for item in data_train:
        user, target = item[0], item[5]
        if user not in user_dict:
            user_dict[user] = [target]
        else:
            user_dict[user] += [target]
    
    for item in user_dict:
        user_dict[item] = np.mean(np.array(user_dict[item]))

    # Predict the user based on his record
    user_not_duplicate_index_list = []
    count = 0

    for i, sample in enumerate(data_test_all):
        if sample[0] in user_dict:
            pred = user_dict[sample[0]]
            output_array[i] = pred
            count += 1
        else:
            user_not_duplicate_index_list.append(i)

    print('|Samples with user who has records: {} samples'.format(count))
    # Predict the user based on nn pre-trained model if he has no record
    user_not_duplicate_list = []

    data_test = [np.load(fp) for fp in data_test_fp_list]

    for index in user_not_duplicate_index_list:
        bulk_index = index // 500000 if index // 500000 <= 3 else 4
        sample = data_test[bulk_index][index % 500000]
        user_not_duplicate_list.append(sample)

    user_not_duplicate_list = np.array(user_not_duplicate_list)
    print('|Samples with user who has no records: {}'.format(len(user_not_duplicate_list)))

    model = NNModel()
    model.load_state_dict(tor.load(model_fp))
    model.eval()

    user_not_duplicate_tsr = tor.FloatTensor(user_not_duplicate_list)
    pred_array = model(user_not_duplicate_tsr).detach().numpy()

    for index, pred in zip(user_not_duplicate_index_list, pred_array):
        output_array[index] = pred

    print('|Prediction size: {}'.format(output_array.shape[0]))

    pd.DataFrame(output_array).to_csv('./prediction_{}_{}.csv'.format(mode, int(time.time())), index_label='id', header=['target'])


    # for test_data_fp in data_test_fp_list:
    #     data = tor.FloatTensor(np.load(test_data_fp).astype(np.float64))
    #     print('|Working on: {}'.format(test_data_fp))
    #     print('|Show test data samples')
    #     print(data[:5]) 

    #     for sample in test_data_fp:
    #         if sample[]
    #     pred = model(data)
    #     pred = pred.detach().numpy().reshape(-1)
    #     print(pred)
    #     pred_array = np.hstack((pred_array, pred))




if __name__ == '__main__':
    mode, model_fp = sys.argv[1], sys.argv[2]

    if mode == 'nn':
        prediction_nn(model_fp)
    elif mode == 'stat':
        prediction_stat(model_fp)