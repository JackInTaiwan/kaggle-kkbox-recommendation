import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as Classifier



def EDA(mode='train'):
    mode = sys.argv[2]

    if mode == 'train':
        data = pd.read_csv('./data/train.csv')
        data = np.array(data)
        source_screen_name = set()
        source_system_tab = set()
        source_type = set()
        # genre_ids = set()
        # language = set()

        for item in data:
            # source_screen_name.add(item['source_screen_name'])
            source_screen_name.add(item[3])
            # source_system_tab.add(item['source_system_tab'])
            source_system_tab.add(item[2])
            # source_type.add(item['source_type'])
            source_type.add(item[4])
            # genre_ids.add(item['genre_ids'])
            # language.add(item['language'])

        print('source_screen_name:', len(source_screen_name))
        print(source_screen_name)
        print('source_system_tab:', len(source_system_tab))
        print(source_system_tab)
        print('source_type', len(source_type))
        print(source_type)
        # print('genre_ids', len(genre_ids))
        # print('language', len(language))

    elif mode == 'song':
        data = pd.read_csv('./data/songs.csv')
        data = np.array(data)

        genre_ids_table = dict()
        language_table = dict()

        for item in data:
            genres = str(item[2]).split('|')
            lan = item[6]
            for genre in genres:
                if genre in genre_ids_table:
                    genre_ids_table[genre] += 1
                else:
                    genre_ids_table[genre] = 1

            if lan in language_table:
                language_table[lan] += 1
            else:
                language_table[lan] = 1
        
        # plt.plot(genre_ids_table.keys(), genre_ids_table.values())
        # plt.show()
        genre_ids_table = sorted(genre_ids_table.items(), key= lambda x: x[1], reverse=True)
        language_table = sorted(language_table.items(), key=lambda x: x[1],reverse=True)

        print('genre_ids:', len(genre_ids_table))
        print(genre_ids_table)
        print('language:', len(language_table))
        print(language_table)

    elif mode == 'user':
        # For the training data users
        data_train = np.array(pd.read_csv('./data/train.csv'))
        
        user_dict = dict()

        for item in data_train:
            user = item[0]
            if user not in user_dict:
                user_dict[user] = 1
            else:
                user_dict[user] += 1
        
        user_array = np.array(sorted(user_dict.values(), reverse=True))
        user_non_repeated = int(np.sum(user_array == 1))

        print('> Training Data <')
        print('|Total unique users: {}'.format(len(user_array)))
        print('|User mean: {}'.format(np.mean(user_array)))
        print('|User 10 pivot values: {}'.format(user_array[[i * len(user_array) // 10 for i in range(10)]]))
        print('|Non repeated user: {:.3f}% {} items'.format(user_non_repeated / len(user_array) * 100, user_non_repeated))

        # For the testing data users
        data_test = np.array(pd.read_csv('./data/test.csv', index_col=0))

        user_test_dict = dict()

        for item in data_test:
            user = item[0]
            if user not in user_test_dict:
                user_test_dict[user] = 1
            else:
                user_test_dict[user] += 1

        user_test_array = np.array(sorted(user_test_dict.values(), reverse=True))
        user_test_non_repeated = int(np.sum(user_test_array == 1))

        print('> Testing Data <')
        print('|Total unique users: {}'.format(len(user_test_array)))
        print('|User mean: {}'.format(np.mean(user_test_array)))
        print('|User 10 pivot values: {}'.format(user_test_array[[i * len(user_test_array) // 10 for i in range(10)]]))
        print('|Non repeated user: {:.3f}% {} items'.format(user_test_non_repeated / len(user_test_array) * 100, user_test_non_repeated))

        # Cross comparison
        user_set = set(user_dict.keys())
        user_test_set = set(user_test_dict.keys())
        user_duplicate_num = len(user_set & user_test_set)
        print('> Cross Comparison <')
        print('|Duplicate users: {:.3f}% {} items'.format(user_duplicate_num / len(user_test_array) * 100, user_duplicate_num))

    elif mode == 'userpattern':
        data_train = np.array(pd.read_csv('./data/train.csv'))
        
        user_dict = dict()

        for item in data_train:
            user, target = item[0], item[5]
            if user not in user_dict:
                user_dict[user] = [target]
            else:
                user_dict[user] += [target]
        
        target_mean_list = sorted(np.array([np.mean(np.array(user_dict[user])) if np.mean(np.array(user_dict[user])) > 0.5 else 1 - np.mean(np.array(user_dict[user])) for user in user_dict.keys()]), reverse=True)
        print('|User pattern 10 pivot: {}'.format([round(target_mean_list[i * len(target_mean_list) // 10], 4) for i in range(10)]))
        print('|User pattern mean: {}'.format(np.mean(target_mean_list)))
        print('|User pattern std: {}'.format(np.std(target_mean_list)))



def select_data(selected_size=10000):
    selected_size = int(sys.argv[2]) if len(sys.argv) > 2 else selected_size
    
    data = pd.read_csv('./data/train.csv')
    print(data.shape)

    data_selected = data.iloc[0:selected_size, :]

    print(data_selected)

    data_selected.to_csv('./data/train_selected.csv')



def data_song_merge():
    data_song = np.array(pd.read_csv('./data/songs.csv'))
    data_song_extra = np.array(pd.read_csv('./data/song_extra_info.csv'))

    song_extra_dict = dict()

    for item in data_song_extra:
        song_extra_dict[item[0][:15]] = item

    new_data_song = []

    for song in data_song:
        new_song = np.hstack((song, [song_extra_dict[song[0][:15]][1]]))
        new_data_song.append(new_song)

    print(len(song_extra_dict))
    new_data_song = sorted(new_data_song, key=lambda x: x[0])
    pd.DataFrame(np.array(new_data_song)).to_csv('./data/songs_all.csv')




def data_merge(bulk_index=0, bulk_size=500000, test_data=False):
    # Params
    test_data, data_fp = bool(int(sys.argv[2])), sys.argv[3]
    if (test_data):
        bulk_index = int(sys.argv[4])
    
    
    # Log
    print('Merge task starts...')
    if (test_data): print('|Test data |index: {}| bulk_size: {}'.format(bulk_index, bulk_size))
    else: print('|Train data')

    
    # Laod data
    data_output_fp = './data/merged_test_data_{}.csv'.format(bulk_index) if test_data else './data/merged_train_data.csv'
    if test_data and bulk_index!=4:
        data = np.array(pd.read_csv(data_fp))[bulk_index * bulk_size: (bulk_index+1) * bulk_size, 1:]
    elif test_data:
        data = np.array(pd.read_csv(data_fp))[bulk_index * bulk_size:, 1:]
    else:
        data = np.array(pd.read_csv(data_fp))[:, 1:]
    data_song = np.array(pd.read_csv('./data/songs_all.csv', encoding='utf-8', index_col=0))
    data_member = np.array(pd.read_csv('./data/members.csv'))

    new_data = np.empty((data.shape[0], 12 if test_data else 13), dtype=object)

    data_song_dict = dict()
    song_dict_key_len = 15
    for song in data_song:
        data_song_dict[song[0][:song_dict_key_len]] = song

    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(i)
        try:
            song = data_song_dict[item[1][:song_dict_key_len]]
        except Exception as e:
            print('song not found', item[1])
            # dummy data
            song = [None, 4 * 60 * 1000, '2163', 'no', None, None, 0, 'no_name']
        try:
            member = data_member[data_member[:, 0] == item[0]][0]
        except Exception as e:
            print('member not found', item[0])
            # dummy data
            member = [None, None, 0, '']
        
        
        try:
            member = member[[2,3]]   # 'bd', 'gender'
            song = song[[1,2,3,6,7]] # 'song_length', 'genre_ids' , 'artist_name', 'language', 'song_name'
            target = item[-1] if not test_data else None
            item = item[:-1]
            new_item = np.hstack((item, member, song)) if test_data else np.hstack((item, member, song, target))
            new_data[i] = new_item
        except Exception as e:
            print(e)
    
    # Ouput the merged file
    new_data = pd.DataFrame(new_data)
    new_data.to_csv(data_output_fp, header=['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'bd', 'gender', 'song_length', 'genre_ids' , 'artist_name', 'language', 'name', 'target'])
    print(new_data.shape)

    # Output the target file
    if not test_data:
        # produce the target file in the case of train data
        output_target_fp = './data/merged_train_target_data.csv'

        new_target_data = data[:, -1]
        pd.DataFrame(new_target_data).to_csv(output_target_fp)
        print(new_target_data.shape)

        
    


def data_preprocess(test_data=False):
    data_fp, test_data = sys.argv[2], bool(int(sys.argv[3]))
    output_fp = './data/prepro_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))

    output = []

    # Labels
    #-- hard code
    hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('nan', 94116), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248), ('125', 18733), ('458', 17857), ('437', 17441), ('444', 16097), ('880', 15430), ('242', 14476), ('451', 13391), ('829', 13155), ('423', 12302), ('2130', 11586), ('2086', 11393), ('1180', 11120), ('1138', 11050), ('2058', 10307), ('374', 8849), ('843', 8413), ('864', 8393), ('850', 8382), ('893', 7778), ('857', 7693), ('430', 7507), ('2072', 7344), ('798', 7104), ('409', 6568), ('352', 5706), ('1995', 4974), ('2107', 4967), ('698', 4851), ('94', 4714), ('545', 4609), ('381', 4469), ('2079', 3924), ('2093', 3881), ('822', 3668), ('1633', 3510), ('1145', 3253), ('118', 2703), ('2189', 2674), ('367', 2154), ('900', 2110)]"
    hit_genre_list = set([item[0] for item in eval(hit_genre_list)])

    bd_pivots = [0, 20, 30, 40, float('inf')]

    source_system_tab = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
    source_system_tab_list = list(eval(source_system_tab))
    print(data[0])
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(i)
        source_system_tab, bd, gender, genre_ids = item[2], item[5], item[6], item[8]
        new_item = []

        # Attri: bd (8)
        try: bd = int(bd)
        except: bd = 0
        if bd == 0:
            new_item += [1/4, 1/4, 1/4, 1/4]
        else:
            new_item += [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]

        # Attri: genre_ids
        if set(str(genre_ids).split('|')) & hit_genre_list:
            new_item += [1]
        else:
            new_item += [0]

        # Attri: source_system_tab
        if source_system_tab in source_system_tab_list:
            new_item += [1 if n == source_system_tab_list.index(source_system_tab) else 0 for n in range(9)]
        else:
            new_item += [0] * 8 + [1]

        # Attri: gender (9)
        if gender == 'female':
            gender = [-1]
        elif gender == 'male':
            gender =  [1]
        else:
            gender = [0]
        new_item += gender
    
        new_item = np.array(new_item, dtype=np.float16)
        output.append(new_item)
    
    output = np.array(output)
    print(output.shape)

    np.save(output_fp, output)  



def EDA_train():
    data_fp, target_fp = sys.argv[2], sys.argv[3]
    data = np.load(data_fp).astype(np.float64)
    target = np.array(pd.read_csv(target_fp, index_col=0))
    x_train = data
    y_train = target.reshape(-1)
    
    print('|Show training samples')
    print('|Train data size: {}'.format(x_train.shape))
    print('|Ratio of 1 in training data: {}'.format(sum(y_train == 1) / y_train.shape[0]))
    print(x_train[:20])
    print(y_train[:20])

    classifier = Classifier(
        hidden_layer_sizes=2 ** 10, activation='relu',
        solver='adam', batch_size=2 ** 3,
        learning_rate_init=10 ** -3,
        tol=1e-6, n_iter_no_change=30,
        max_iter=20, shuffle=True,
        verbose=True
    )

    classifier.fit(x_train, y_train)

    pred = classifier.predict(x_train)
    score = classifier.score(x_train, y_train)
    print(score)

    joblib.dump(classifier, './models/model_{}.pkl'.format(time.time()))
    
    


def EDA_pred():
    model_fp = sys.argv[2]
    test_data_fp_list = ['./data/prepro_merged_test_data_{}.npy'.format(i) for i in range(5)]

    model = joblib.load(model_fp)
    pred_array = np.array([])

    for test_data_fp in test_data_fp_list:
        data = np.load(test_data_fp).astype(np.float64)
        print('|Working on: {}'.format(test_data_fp))
        print('|Show test data samples')
        print(data[:5]) 

        pred = model.predict_proba(data)[:, 1].reshape(-1)
        pred_array = np.hstack((pred_array, pred))
    
    pred_array = pred_array.reshape(-1, 1)
    print('|Prediction size: {}'.format(pred_array.shape[0]))

    pd.DataFrame(pred_array).to_csv('./prediction_{}.csv'.format(int(time.time())),index_label='id', header=['target'])






if __name__ == '__main__':
    task_index = int(sys.argv[1])

    if task_index == 1:
        EDA('song')
    elif task_index == 2:
        select_data()
    elif task_index == 3:
        data_song_merge()
    elif task_index == 4:
        data_merge()
    elif task_index == 5:
        data_preprocess()
    elif task_index == 6:
        EDA_train()
    elif task_index == 7:
        EDA_pred()