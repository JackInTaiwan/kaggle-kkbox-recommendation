import sys
import numpy as np
import pandas as pd
import pickle



def preprocess_memory(is_train_data, data_fp):
    def sample_encode(sample, is_train=True):
        # Labels
        hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248)]"
        hit_genre_list = set([item[0] for item in eval(hit_genre_list)])

        bd_pivots = [0, 20, 30, 40, float('inf')]

        source_system_tab = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
        source_system_tab_list = list(eval(source_system_tab))

        song_lan_list = "[(52.0, 1336731), (-1.0, 639481), (3.0, 106301)]"
        song_lan_list = [item[0] for item in eval(song_lan_list)]
        source_system_tab, bd, gender, genre_ids, language = sample[2], sample[5], sample[6], sample[8], sample[10]
        target = sample[-1] if is_train else None
        new_sample = np.empty((4 + 1 + 9 + 1 + 4 + 1)) if is_train else np.empty((4 + 1 + 9 + 1 + 4))

        # Attri: bd (4)
        try: bd = int(bd)
        except: bd = 0
        if bd == 0:
            new_sample[0:4] = [1/4, 1/4, 1/4, 1/4]
        else:
            new_sample[0:4] = [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]

        # Attri: gender
        if gender == 'female':
            gender = -1
        elif gender == 'male':
            gender = 1
        else:
            gender = 0
        new_sample[4] = gender
        
        # Attri: genre_ids (1)
        if set(genre_ids.split('|')) & hit_genre_list:
            new_sample[5] = 1
        else:
            new_sample[5] = 0

        # Attri: source_system_tab (9)
        if source_system_tab in source_system_tab_list:
            new_sample[6:15] = [1 if n == source_system_tab_list.index(source_system_tab) else 0 for n in range(9)]
        else:
            new_sample[6:15] = [0] * 8 + [1]

        # Attri: song language (4)
        if language in song_lan_list:
            new_sample[15:19] = [1 if n == song_lan_list.index(language) else 0 for n in range(len(song_lan_list)+1)]
        else: 
            new_sample[15:19] = ([0] * len(song_lan_list) + [1])

        # Attri: target
        if is_train:
            new_sample[19] = target

        new_sample = np.array(new_sample, dtype=np.float16)

        return new_sample

    ### Load data   
    output_fp = './data/prepro_memory_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))

    print('|Start building dict...')
    if is_train_data:
        data_dict = dict()
        for item in data:
            if item[0] not in data_dict:
                data_dict[item[0]] = np.array([item])
            else:
                data_dict[item[0]] = np.vstack((data_dict[item[0]], item))
    else:
        data_train_fp = './data/merged_train_selected_data.csv'
        data_train = np.array(pd.read_csv(data_train_fp, index_col=0))
        data_dict = dict()
        for item in data_train:
            if item[0] not in data_dict:
                data_dict[item[0]] = np.array([item])
            else:
                data_dict[item[0]] = np.vstack((data_dict[item[0]], item))


    ### Preprocess for training data
    print('|Start preprocess...')
    if is_train_data:
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
        np.save(output_fp, output)
        
    ### Preprocess for testing data
    else:
        output = []

        keys_to_be_del = []
        for key, data_array in data_dict.items():
            if (sum(data_array[:, -1] == 1) >= 2 and sum(data_array[:, -1] == 0) >= 2):
                new_data_array = np.array([sample_encode(item) for item in data_array])
                base_target_sample_array = new_data_array[new_data_array[:, -1] == 1]
                base_nontarget_sample_array = new_data_array[new_data_array[:, -1] == 0]
                base_target_sample_avg = np.mean(base_target_sample_array[:, 5:-1], axis=0)
                base_nontarget_sample_avg = np.mean(base_nontarget_sample_array[:, 5:-1], axis=0)
                target_ratio = np.array([sum(new_data_array[:, -1] == 1) / len(new_data_array)])
                user_info = new_data_array[0][0:5]
                data_dict[key] = np.hstack((user_info, base_target_sample_avg, base_nontarget_sample_avg, target_ratio))
            else:
                keys_to_be_del.append(key)

        for key in keys_to_be_del:
            del data_dict[key]

        count_no_base = 0
        for sample in data:
            if sample[0] in data_dict:
                base = data_dict[sample[0]]
                sample = sample_encode(sample, is_train=False)
                new_sample = np.hstack((sample[5:], base, np.array([1])))   # 1 denotes it has base
                output.append(new_sample)
            else:
                sample = sample_encode(sample, is_train=False)
                new_sample = np.hstack((sample[5:], sample[0:5], np.array([0.5] * 29), np.array([0])))   # 1 denotes it has base
                output.append(new_sample)
                count_no_base += 1

        output = np.array(output)
        np.save(output_fp, output)



def preprocess_dt(is_train_data, data_fp, dict_fp):
    def sample_encode(sample, pos_base_list, neg_base_list, is_train=True):
        new_sample = np.empty((4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1), dtype=np.float16) if is_train else np.empty((4 + 1 + 1 + 1 + 1 + 1 + 1 + 1), dtype=np.float16)
        
        source_system_tab, bd, gender, genre_ids_set, artist_name, language = sample[2], sample[5], sample[6], sample[8], sample[9], sample[10]
        target = sample[-1] if is_train else None
        
        # Attri: bd [0:4]
        bd_pivots = [0, 20, 30, 40, float('inf')]
        try: bd = int(bd)
        except: bd = 0
        if bd == 0:
            new_sample[0:4] = [1/4, 1/4, 1/4, 1/4]
        else:
            new_sample[0:4] = [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]
        
        # Attri: is duplicate song genre_id [4]
        duplicate_count = 0
        for pos_base in pos_base_list:
            if genre_ids_set & pos_base[8]:
                duplicate_count += 1
        
        new_sample[4] = 1 if duplicate_count else 0
        
        # Attri: ratio of duplicate song genre_id [5]
        duplicate_ratio = duplicate_count / len(pos_base_list) if len(pos_base_list) != 0 else 0
        new_sample[5] = duplicate_ratio

        # Attri: is a hit genre_ids [6]
        hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248)]"
        hit_genre_list = set([item[0] for item in eval(hit_genre_list)])
        if genre_ids_set & hit_genre_list:
            new_sample[6] = 1
        else:
            new_sample[6] = 0

        # Attri: is duplicate song language [7]
        duplicate_count = 0
        for pos_base in pos_base_list:
            if (language == pos_base[10]):
                new_sample[7] = 1
                break
        else:
            new_sample[7] = 0

        # Attri: score of source_system_tab [8]
        source_system_tab_list = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
        source_system_tab_list = list(eval(source_system_tab_list))
        source_pos_v = np.zeros((9))
        source_neg_v = np.zeros((9))
        for pos_base in pos_base_list:
            index = source_system_tab_list.index(pos_base[2]) if pos_base[2] in source_system_tab_list else 8
            source_pos_v[index] += 1
        for neg_base in neg_base_list:
            index = source_system_tab_list.index(neg_base[2]) if neg_base[2] in source_system_tab_list else 8
            source_neg_v[index] += 1
        source_pos_v = source_pos_v / len(pos_base_list) if len(pos_base_list) != 0 else source_pos_v
        source_neg_v = source_neg_v / len(neg_base_list) if len(neg_base_list) != 0 else source_neg_v

        index_sample_source = source_system_tab_list.index(source_system_tab) if source_system_tab in source_system_tab_list else 8
        source_score = (source_pos_v - source_neg_v)[index_sample_source]
        new_sample[8] = source_score

        # Attri: is duplicate artist_name
        for pos_base in pos_base_list:
            if (artist_name == pos_base[9]):
                new_sample[9] = 1
                break
        else:
            new_sample[9] = 0

        # Attri: target ratio (train data only) [10]
        new_sample[10] = len(pos_base_list) / (len(pos_base_list) + len(neg_base_list))

        # Attri: target (train data only) [11]
        if is_train:
            new_sample[11] = target

        new_sample = np.array(new_sample, dtype=np.float16)
        return new_sample

    
    ### Load data
    output_fp = './data/prepro_final_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))


    ### Build/Load data dictionary
    if not dict_fp:
        print('|Start building dict...', flush=True)
        if is_train_data:
            data_dict = dict()
            for item in data:
                if item[0] not in data_dict:
                    data_dict[item[0]] = np.array([item])
                else:
                    data_dict[item[0]] = np.vstack((data_dict[item[0]], item))
        else:
            data_train_fp = './data/merged_train_data.csv'
            data_train = np.array(pd.read_csv(data_train_fp, index_col=0))
            data_dict = dict()
            for item in data_train:
                if item[0] not in data_dict:
                    data_dict[item[0]] = np.array([item])
                else:
                    data_dict[item[0]] = np.vstack((data_dict[item[0]], item))
        with open('./data/merged_train_data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            print('save dict !', flush=True)
    else:
        print('|Start loading dict...', flush=True)
        with open(dict_fp, 'rb') as f:
            data_dict = pickle.load(f)

    if np.nan in data_dict:
        print('use delete')
        del data_dict[np.nan]


    ### Preprocess for training data
    print('|Start preprocess...', flush=True)
    if is_train_data:
        output = np.empty((data.shape[0], 11 + 1))
        data_lose_count = 0

        count_total = 0
        for data_array in data_dict.values():
            if (sum(data_array[:, -1] == 1) >= 2 and sum(data_array[:, -1] == 0) >= 2):
                for i, datum in enumerate(data_array):
                    datum[8] = set(str(datum[8]).split('|'))
                    data_array[i] = datum
                    
                for i, sample in enumerate(data_array):
                    if count_total % 100000 == 0:
                        print('|Round: ', count_total, flush=True)
                    data_base_array = np.delete(data_array, i, 0)
                    base_target_sample_array = data_base_array[data_base_array[:, -1] == 1]
                    base_nontarget_sample_array = data_base_array[data_base_array[:, -1] == 0]
                    new_sample = sample_encode(sample, base_target_sample_array, base_nontarget_sample_array)
                    # output.append(new_sample)
                    output[count_total] = new_sample
                    count_total += 1
            else:
                data_lose_count += data_array.shape[0]

        output = np.array(output[:count_total])
        np.save(output_fp, output)

    ### Preprocess for testing data
    else:
        output = np.empty((data.shape[0], 11 + 1))

        keys_to_be_del = []
        for key, data_array in data_dict.items():
            if (len(data_array) > 1):
                for i, datum in enumerate(data_array):
                    datum[8] = set(str(datum[8]).split('|'))
                    data_array[i] = datum
            else:
                keys_to_be_del.append(key)

        for key in keys_to_be_del:
            del data_dict[key]

        count_no_base = 0
        for i, sample in enumerate(data):
            if i % 100000 == 0:
                print('|Round: ', i)
            if sample[0] in data_dict:
                sample[8] = set(str(sample[8]).split('|'))
                data_array = data_dict[sample[0]]
                base_target_sample_array = data_array[data_array[:, -1] == 1]
                base_nontarget_sample_array = data_array[data_array[:, -1] == 0]
                new_sample = sample_encode(sample, base_target_sample_array, base_nontarget_sample_array, is_train=False)
                new_sample = np.hstack((new_sample, 1))
                output[i] = new_sample
            else:
                new_sample = np.hstack((np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5]), 0))
                output[i] = new_sample
                count_no_base += 1

        output = np.array(output)
        np.save(output_fp, output)

    print('|Task Done', flush=True)



def preprocess_extend(is_train_data, data_fp, dict_fp):
    def sample_encode(sample, pos_base_list, neg_base_list, index=None, is_train=True):
        new_sample = np.empty((2 + 17 + 1), dtype=np.object) if is_train else np.empty((2 + 17), dtype=np.object)
        
        msno, source_system_tab, bd, gender, genre_ids_set, artist_name, language = sample[0], sample[2], sample[5], sample[6], sample[8], sample[9], sample[10]
        target = sample[-1] if is_train else None
        
        # Attr: index [0]
        new_sample[0] = index

        # Attr: msno [1]
        new_sample[1] = msno

        # Attr: bd [2:6]
        bd_pivots = [0, 20, 30, 40, float('inf')]
        try: bd = int(bd)
        except: bd = 0
        if bd == 0:
            new_sample[2:6] = [1/4, 1/4, 1/4, 1/4]
        else:
            new_sample[2:6] = [1 if bd_pivots[n-1]< bd <= bd_pivots[n] else 0 for n in range(1, 5)]
        
        # Attr: is duplicate song genre_id on pos_base_list [6]
        duplicate_count = 0
        for pos_base in pos_base_list:
            if genre_ids_set & pos_base[8]:
                duplicate_count += 1
        new_sample[6] = 1 if duplicate_count else 0

        # Attr: ratio of duplicate song genre_id [7]
        duplicate_ratio = duplicate_count / len(pos_base_list) if len(pos_base_list) != 0 else 0
        new_sample[7] = duplicate_ratio

        # Attr: is duplicate song genre_id on neg_base_list [8]
        duplicate_count = 0
        for neg_base in neg_base_list:
            if genre_ids_set & neg_base[8]:
                duplicate_count += 1
        new_sample[8] = 1 if duplicate_count else 0
        
        # Attr: ratio of duplicate song genre_id on neg_base_list [9]
        duplicate_ratio = duplicate_count / len(pos_base_list) if len(pos_base_list) != 0 else 0
        new_sample[9] = duplicate_ratio

        # Attr: std of duplicate song genre_id on post_base_list [10]
        song_genre_dict = dict()
        count_total = 0
        for pos_base in pos_base_list:
            for genre in pos_base[8]:
                count_total += 1
                if genre in song_genre_dict:
                    song_genre_dict[genre] += 1
                else:
                    song_genre_dict[genre] = 1
        genre_std = np.std(np.array(list(song_genre_dict.values())) / count_total) if count_total > 0 else 0
        new_sample[10] = genre_std

        # Attr: the diff of duplicate song genre_id between pos_base_list and neg_base_list [11]
        genre_diff = new_sample[7] - new_sample[9]
        new_sample[11] = genre_diff

        # Attri: is a hit genre_ids [12]
        hit_genre_list = "[('465', 589220), ('958', 182836), ('1609', 177258), ('2022', 176531), ('2122', 149608), ('1259', 103904), ('921', 74983), ('1152', 65463), ('786', 59438), ('139', 56405), ('359', 48144), ('940', 45604), ('726', 36766), ('1011', 34620), ('947', 30232), ('388', 27608), ('1572', 27311), ('1616', 26983), ('275', 25808), ('1955', 21426), ('109', 20659), ('873', 20513), ('691', 20248)]"
        hit_genre_list = set([item[0] for item in eval(hit_genre_list)])
        if genre_ids_set & hit_genre_list:
            new_sample[12] = 1
        else:
            new_sample[12] = 0

        # Attri: is duplicate song language on pos_base_list [13]
        duplicate_count = 0
        for pos_base in pos_base_list:
            if (language == pos_base[10]):
                new_sample[13] = 1
                break
        else:
            new_sample[13] = 0

        # Attri: is duplicate song language on neg_base_list [14]
        duplicate_count = 0
        for neg_base in neg_base_list:
            if (language == neg_base[10]):
                new_sample[14] = 1
                break
        else:
            new_sample[14] = 0

        # Attri: score of source_system_tab [15]
        source_system_tab_list = "{'explore', 'search', 'notification', 'discover', 'listen with', 'radio', 'my library', 'settings'}"
        source_system_tab_list = list(eval(source_system_tab_list))
        source_pos_v = np.zeros((9))
        source_neg_v = np.zeros((9))
        for pos_base in pos_base_list:
            index = source_system_tab_list.index(pos_base[2]) if pos_base[2] in source_system_tab_list else 8
            source_pos_v[index] += 1
        for neg_base in neg_base_list:
            index = source_system_tab_list.index(neg_base[2]) if neg_base[2] in source_system_tab_list else 8
            source_neg_v[index] += 1
        source_pos_v = source_pos_v / len(pos_base_list) if len(pos_base_list) != 0 else source_pos_v
        source_neg_v = source_neg_v / len(neg_base_list) if len(neg_base_list) != 0 else source_neg_v

        index_sample_source = source_system_tab_list.index(source_system_tab) if source_system_tab in source_system_tab_list else 8
        source_score = (source_pos_v - source_neg_v)[index_sample_source]
        new_sample[15] = source_score

        # Attri: is duplicate artist_name [16]
        # Attri: ratio of duplicate artist_name on pos_base_list [17]
        in_post, in_neg = 0, 0
        count_in_post = 0
        for pos_base in pos_base_list:
            if artist_name == pos_base[9]:
                in_post = 1
                count_in_post += 1
        for neg_base in neg_base_list:
            if artist_name == neg_base[9]:
                in_neg = 1
                break
        new_sample[16] = in_post - in_neg
        new_sample[17] = count_in_post / len(pos_base_list) if len(pos_base_list) > 0 else 0

        # Attri: target ratio (train data only) [18]
        new_sample[18] = len(pos_base_list) / (len(pos_base_list) + len(neg_base_list))

        # Attri: target (train data only) [19]
        if is_train:
            new_sample[19] = target

        new_sample = new_sample if is_train else new_sample[1:]
        return new_sample

    
    ### Load data
    output_fp = './data/prepro_extend_{}'.format(data_fp.split('/')[-1].split('.')[0])
    data = np.array(pd.read_csv(data_fp, index_col=0))


    ### Build/Load data dictionary
    if not dict_fp:
        print('|Start building dict...', flush=True)
        if is_train_data:
            data_dict = dict()
            for item in data:
                if item[0] not in data_dict:
                    data_dict[item[0]] = np.array([item])
                else:
                    data_dict[item[0]] = np.vstack((data_dict[item[0]], item))
        else:
            data_train_fp = './data/merged_train_data.csv'
            data_train = np.array(pd.read_csv(data_train_fp, index_col=0))
            data_dict = dict()
            for item in data_train:
                if item[0] not in data_dict:
                    data_dict[item[0]] = np.array([item])
                else:
                    data_dict[item[0]] = np.vstack((data_dict[item[0]], item))
        if np.nan in data_dict: del data_dict[np.nan]
        
        with open('./data/merged_train_selected_data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            print('|Saving dict succeed.', flush=True)
    else:
        print('|Start loading dict...', flush=True)
        with open(dict_fp, 'rb') as f:
            data_dict = pickle.load(f)


    ### Preprocess for training data
    print('|Start preprocess...', flush=True)
    if is_train_data:
        output = np.empty((data.shape[0], 2 + 17 + 1), dtype=np.object)
        data_lose_count = 0
        count_total = 0

        for index_cluster, data_array in enumerate(data_dict.values()):
            if (len(data_array) > 1):
                for i, datum in enumerate(data_array):
                    datum[8] = set(str(datum[8]).split('|'))
                    data_array[i] = datum
                    
                for i, sample in enumerate(data_array):
                    if count_total % 100000 == 0:
                        print('|Round: ', count_total, flush=True)
                    data_base_array = np.delete(data_array, i, 0)
                    base_target_sample_array = data_base_array[data_base_array[:, -1] == 1]
                    base_nontarget_sample_array = data_base_array[data_base_array[:, -1] == 0]
                    new_sample = sample_encode(sample, base_target_sample_array, base_nontarget_sample_array, index_cluster, is_train=True)
                    output[count_total] = new_sample
                    count_total += 1
            else:
                data_lose_count += data_array.shape[0]

        output = np.array(output[:count_total])
        np.save(output_fp, output)

    ### Preprocess for training data
    else:
        output = np.empty((data.shape[0], 1 + 17 + 1), dtype=np.object)
        keys_to_be_del = []

        for key, data_array in data_dict.items():
            if (len(data_array) > 1):
                for i, datum in enumerate(data_array):
                    datum[8] = set(str(datum[8]).split('|'))
                    data_array[i] = datum
            else:
                keys_to_be_del.append(key)

        for key in keys_to_be_del:
            del data_dict[key]

        count_no_base = 0
        for i, sample in enumerate(data):
            if i % 100000 == 0:
                print('|Round: ', i)
            if sample[0] in data_dict:
                sample[8] = set(str(sample[8]).split('|'))
                data_array = data_dict[sample[0]]
                base_target_sample_array = data_array[data_array[:, -1] == 1]
                base_nontarget_sample_array = data_array[data_array[:, -1] == 0]
                new_sample = sample_encode(sample, base_target_sample_array, base_nontarget_sample_array, is_train=False)
                new_sample = np.hstack((new_sample, 1))
                output[i] = new_sample
            else:
                new_sample = np.hstack((np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5]), 0))
                output[i] = new_sample
                count_no_base += 1

        np.save(output_fp, output)

    print('|Task Done', flush=True)
    


if __name__ == '__main__':
    mode, is_train_data, data_fp, dict_fp = sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], sys.argv[4] if len(sys.argv) >=5 else None
    
    if mode == 'memory':
        preprocess_memory(is_train_data, data_fp)
    elif mode == 'dt':
        preprocess_dt(is_train_data, data_fp, dict_fp)
    elif mode == 'extend':
        preprocess_extend(is_train_data, data_fp, dict_fp)