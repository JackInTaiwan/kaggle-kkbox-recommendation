import sys
import time
import numpy as np
import pandas as pd
import torch as tor



def prediction_simple_nn(model_fp):
    ### Load data
    data_test_fp_list = ['./data/prepro_merged_test_data_{}.npy'.format(i) for i in range(5)]

    model = NNModel()
    model.load_state_dict(tor.load(model_fp))
    model.eval()


    ### Prediction
    print('|Start prediction......'
    )
    pred_array = np.array([])
    for test_data_fp in data_test_fp_list:
        data = tor.FloatTensor(np.load(test_data_fp).astype(np.float64))
        print('|Working on: {}'.format(test_data_fp))
        print('|Show test data samples')

        pred = model(data)
        pred = pred.detach().numpy().reshape(-1)
        pred_array = np.hstack((pred_array, pred))
    
    pred_array = pred_array.reshape(-1, 1)
    pd.DataFrame(pred_array).to_csv('./prediction_{}_{}.csv'.format(mode, int(time.time())), index_label='id', header=['target'])

    print('|Prediction size: {}'.format(pred_array.shape[0]))



def prediction_stat(model_fp):
    from train import NNModel

    ### Load data
    data_train = np.array(pd.read_csv('./data/train.csv'))
    data_test_all_fp = './data/test.csv'
    data_test_fp_list = ['./data/prepro_merged_test_data_{}.npy'.format(i) for i in range(5)]
    data_test_all = np.array(pd.read_csv(data_test_all_fp, index_col=0))
    output_array = np.empty((len(data_test_all)))


    ### Build user dict
    user_dict = dict()

    for item in data_train:
        user, target = item[0], item[5]
        if user not in user_dict:
            user_dict[user] = [target]
        else:
            user_dict[user] += [target]
    
    for item in user_dict:
        user_dict[item] = np.mean(np.array(user_dict[item]))


    ### Predict the user based on his record
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

    user_not_duplicate_list = []
    data_test = [np.load(fp) for fp in data_test_fp_list]

    for index in user_not_duplicate_index_list:
        bulk_index = index // 500000 if index // 500000 <= 3 else 4
        sample = data_test[bulk_index][index % 500000]
        user_not_duplicate_list.append(sample)

    user_not_duplicate_list = np.array(user_not_duplicate_list)
    print('|Samples with user who has no records: {}'.format(len(user_not_duplicate_list)))


    ### Prediction
    print('|Start prediction......')
    model = NNModel()
    model.load_state_dict(tor.load(model_fp))
    model.eval()

    user_not_duplicate_tsr = tor.FloatTensor(user_not_duplicate_list)
    pred_array = model(user_not_duplicate_tsr).detach().numpy()

    for index, pred in zip(user_not_duplicate_index_list, pred_array):
        output_array[index] = pred
    pd.DataFrame(output_array).to_csv('./prediction_{}_{}.csv'.format(mode, int(time.time())), index_label='id', header=['target'])

    print('|Prediction size: {}'.format(output_array.shape[0]))



def prediction_memory(model_fp):
    from train import Memory_Based_Trainer
    from torch.utils.data import TensorDataset, DataLoader


    ### Load data and model
    data_test_fp = './data/prepro_memory_merged_test_data.npy'
    data_test = np.load(data_test_fp)
    data_test = data_test[:, :-1]   # remove the indicator of existence of base

    data_set = TensorDataset(tor.FloatTensor(data_test))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
    )

    model = Memory_Based_Trainer.create_model(model_name='ratio_base_modified')
    model.load_state_dict(tor.load(model_fp))
    model.cpu()
    model.eval()


    ### Prediction
    print('|Start prediction ...')

    output_array = np.array([])
    for i, batch in enumerate(data_loader):
        batch_test = batch[0]
        pred_array = model(batch_test).detach().numpy()
        if output_array.shape[0] == 0:
            output_array = pred_array
        else:
            output_array = np.vstack((output_array, pred_array))

    output_array = np.array(output_array)
    pd.DataFrame(output_array).to_csv('./predictions/prediction_{}_{}.csv'.format(mode, int(time.time())), index_label='id', header=['target'])

    print('|Prediction size: {}'.format(output_array.shape[0]))



if __name__ == '__main__':
    mode, model_fp = sys.argv[1], sys.argv[2]

    if mode == 'nn':
        prediction_simple_nn(model_fp)
    elif mode == 'stat':
        prediction_stat(model_fp)
    elif mode == 'memory':
        prediction_memory(model_fp)