import sys
import numpy as np
import pandas as pd



def preprocess(data_fp):
    output_fp = './data/prepro_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))

    output = []

    # Labels
    #-- hard code
    # hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248), ('125', 18733), ('458', 17857), ('437', 17441), ('444', 16097), ('880', 15430), ('242', 14476), ('451', 13391), ('829', 13155), ('423', 12302), ('2130', 11586), ('2086', 11393), ('1180', 11120), ('1138', 11050), ('2058', 10307), ('374', 8849), ('843', 8413), ('864', 8393), ('850', 8382), ('893', 7778), ('857', 7693), ('430', 7507), ('2072', 7344), ('798', 7104), ('409', 6568), ('352', 5706), ('1995', 4974), ('2107', 4967), ('698', 4851), ('94', 4714), ('545', 4609), ('381', 4469), ('2079', 3924), ('2093', 3881), ('822', 3668), ('1633', 3510), ('1145', 3253), ('118', 2703), ('2189', 2674), ('367', 2154), ('900', 2110)]"
    hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248)]"
    hit_genre_list = set([item[0] for item in eval(hit_genre_list)])

    bd_pivots = [0, 20, 30, 40, float('inf')]

    source_system_tab = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
    source_system_tab_list = list(eval(source_system_tab))

    song_lan_list = "[(52.0, 1336731), (-1.0, 639481), (3.0, 106301)]"
    song_lan_list = [item[0] for item in eval(song_lan_list)]

    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(i)
        source_system_tab, bd, gender, genre_ids, language = item[2], item[5], item[6], item[8], item[10]
        new_item = np.empty((4 + 1 + 9 + 1 + 4))

        # Attri: bd (4)
        try: bd = int(bd)
        except: bd = 0
        if bd == 0:
            new_item[0:4] = [1/4, 1/4, 1/4, 1/4]
        else:
            new_item[0:4] += [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]
        
        # Attri: genre_ids (1)
        if set(str(genre_ids).split('|')) & hit_genre_list:
            new_item[4] = 1
        else:
            new_item[4] = 0

        # Attri: source_system_tab (9)
        if source_system_tab in source_system_tab_list:
            new_item[5:14] = [1 if n == source_system_tab_list.index(source_system_tab) else 0 for n in range(9)]
        else:
            new_item[5:14] = [0] * 8 + [1]

        if gender == 'female':
            gender = -1
        elif gender == 'male':
            gender = 1
        else:
            gender = 0
        new_item[14] = gender

        # Attri: song language (4)
        if language in song_lan_list:
            new_item[15:19] = [1 if n == song_lan_list.index(language) else 0 for n in range(len(song_lan_list)+1)]
        else: 
            new_item[15:19] = ([0] * len(song_lan_list) + [1])

    
        new_item = np.array(new_item, dtype=np.float16)
        output.append(new_item)
    
    output = np.array(output)
    print(output.shape)

    np.save(output_fp, output)  


def sample_encode(sample):
    # Labels
    #-- hard code
    hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248)]"
    hit_genre_list = set([item[0] for item in eval(hit_genre_list)])

    bd_pivots = [0, 20, 30, 40, float('inf')]

    source_system_tab = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
    source_system_tab_list = list(eval(source_system_tab))

    song_lan_list = "[(52.0, 1336731), (-1.0, 639481), (3.0, 106301)]"
    song_lan_list = [item[0] for item in eval(song_lan_list)]
    source_system_tab, bd, gender, genre_ids, language, target = sample[2], sample[5], sample[6], sample[8], sample[10], sample[-1]
    new_sample = np.empty((4 + 1 + 9 + 1 + 4 + 1))

    # Attri: bd (4)
    try: bd = int(bd)
    except: bd = 0
    if bd == 0:
        new_sample[0:4] = [1/4, 1/4, 1/4, 1/4]
    else:
        new_sample[0:4] += [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]
    
    # Attri: genre_ids (1)
    if set(str(genre_ids).split('|')) & hit_genre_list:
        new_sample[4] = 1
    else:
        new_sample[4] = 0

    # Attri: source_system_tab (9)
    if source_system_tab in source_system_tab_list:
        new_sample[5:14] = [1 if n == source_system_tab_list.index(source_system_tab) else 0 for n in range(9)]
    else:
        new_sample[5:14] = [0] * 8 + [1]

    if gender == 'female':
        gender = -1
    elif gender == 'male':
        gender = 1
    else:
        gender = 0
    new_sample[14] = gender

    # Attri: song language (4)
    if language in song_lan_list:
        new_sample[15:19] = [1 if n == song_lan_list.index(language) else 0 for n in range(len(song_lan_list)+1)]
    else: 
        new_sample[15:19] = ([0] * len(song_lan_list) + [1])

    new_sample[19] = target

    new_sample = np.array(new_sample, dtype=np.float16)

    return new_sample


def preprocess_memory(is_train_data, data_fp):
    output_fp = './data/prepro_memory_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))

    data_dict = dict()
    for item in data:
        if item[0] not in data_dict:
            data_dict[item[0]] = np.array([item])
        else:
            data_dict[item[0]] = np.vstack((data_dict[item[0]], item))

    output = []
    data_lose_count = 0
    
    for data_array in data_dict.values():
        if (sum(data_array[:, -1] == 1) >= 2 and sum(data_array[:, -1] == 0) >= 2):
            target_ratio = sum(data_array[:, -1] == 1) / len(data_array)

            new_sample_dim = 20
            new_data_array = np.empty((data_array.shape[0], new_sample_dim))
            for i, sample in enumerate(data_array):
                new_sample = sample_encode(sample)
                new_data_array[i]= new_sample
                
            for i, sample in enumerate(new_data_array):
                base_sample_array = np.delete(new_data_array, i, 0)
                base_target_sample_array = base_sample_array[base_sample_array[:, -1] == 1]
                base_nontarget_sample_array = base_sample_array[base_sample_array[:, -1] == 0]
                bd, gender = base_target_sample_array[0, 0:4], base_target_sample_array[0, 4:5]

                base_target_sample_avg = np.mean(base_target_sample_array[:, 5:-1], axis=0)
                base_nontarget_sample_avg = np.mean(base_nontarget_sample_array[:, 5:-1], axis=0)
                new_sample = np.hstack((sample[5:-1], bd, gender, base_target_sample_avg, base_nontarget_sample_avg, target_ratio, sample[-1]))

                output.append(new_sample)
        else:
            data_lose_count += data_array.shape[0]

    output = np.array(output)
    print(output.shape)
    print(data_lose_count)

    np.save(output_fp, output)  


if __name__ == '__main__':
    mode, is_train_data, data_fp = sys.argv[1], sys.argv[2], sys.argv[3]
    
    if mode == 'simple':
        preprocess(data_fp)
    elif mode == 'memory':
        preprocess_memory(is_train_data, data_fp)