import sys
import time
import pandas as pd
import torch as tor
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from itertools import product



class Simple_NN_Trainer(data_fp, target_fp):
    @staticmethod
    def create_model():
        class NNModel(tor.nn.Module):
            def __init__(self):
                super(NNModel, self).__init__()

                self.fc_1 = tor.nn.Linear(19, 2 ** 9)
                self.fc_2 = tor.nn.Linear(2 ** 9, 2 ** 9)
                self.fc_3 = tor.nn.Linear(2 ** 9, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                o = self.fc_1(x)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.fc_3(o)
                output = self.sig(o) 

                return output
        
        return NNModel()


    ### Load date
    x_train = np.load(data_fp)
    y_train = np.array(pd.read_csv(target_fp, index_col=0))
    
    data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=2 ** 3,
        shuffle=True,
        drop_last=True,
    )
    

    ### Model params
    nn = create_model()
    loss_func = tor.nn.MSELoss()
    optim = tor.optim.Adam(nn.parameters(), lr=10 ** -4)


    ### Training
    for epoch in range(30):
        print('|Epoch: ', epoch + 1)
        total_loss = 0
        for step, (x, y) in enumerate(data_loader):
            optim.zero_grad()
            pred = nn(x)
            
            loss = loss_func(pred, y)
            total_loss += loss
            loss.backward()
            optim.step()

            if step % 2000 == 0:
                print('|Loss:', float(total_loss / 1000))
                total_loss = 0

            if step % 10000 == 0:
                nn.eval()
                pred = nn(tor.FloatTensor(x_train)).detach()
                y_train = tor.FloatTensor(y_train)
                acc = tor.mean(tor.tensor((pred > 0.5) == (y_train == 1), dtype=tor.float), dim=0)
                print('|Acc: ', float(acc))
                nn.train()

        ### Validation loss
        optim.zero_grad()
        tor.save(nn.state_dict(), './models/model_nn_{}.pkl'.format(int(time.time())))



class Memory_Based_Trainer():
    def __init__(self, argv):
        self.data_fp = argv[0]
        self.gpu = bool(int(argv[1]))
        self.lr = 10 ** -4
        self.batch_size = 2 ** 3
        self.start_time = int(time.time())

    @staticmethod
    def create_model(model_name='nn'):
        class NN(tor.nn.Module):
            def __init__(self):
                super(NN, self).__init__()
                # input shape: (48) = (14, 5. 14, 14, 1) = (sample, userinfo, target_sample_avg, nontarget_sample_avg, target_ratio)
                self.fc_1 = tor.nn.Linear(5 + 14 + 14 + 1, 2 ** 10)
                self.fc_2 = tor.nn.Linear(2 ** 10, 2 ** 10)
                self.fc_3 = tor.nn.Linear(2 ** 10, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                sample, userinfo, target_sample_avg, nontarget_sample, target_ratio = x[:, 0:14], x[:, 14:19], x[:, 19:33], x[:, 33:47], x[:, 47:48]
                x = tor.cat((userinfo, tor.abs(sample - target_sample_avg), tor.abs(sample - nontarget_sample), target_ratio), 1)
                o = self.fc_1(x)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.fc_3(o)
                output = self.sig(o) 

        class Ratio_Base_Mode(tor.nn.Module):
            def __init__(self):
                super(Ratio_Base_Mode, self).__init__()
                # input shape: (48) = (14, 5. 14, 14, 1) = (sample, userinfo, target_sample_avg, nontarget_sample_avg, target_ratio)
                self.fc_1 = tor.nn.Linear(5 + 14 + 14 + 1, 2 ** 10)
                self.fc_2 = tor.nn.Linear(2 ** 10, 2 ** 10)
                self.fc_3 = tor.nn.Linear(2 ** 10, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.tanh = tor.nn.Tanh()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                sample, userinfo, target_sample_avg, nontarget_sample, target_ratio = x[:, 0:14], x[:, 14:19], x[:, 19:33], x[:, 33:47], x[:, 47:48]
                x = tor.cat((userinfo, tor.abs(sample - target_sample_avg), tor.abs(sample - nontarget_sample), target_ratio), 1)
                o = self.fc_1(x)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.tanh(self.fc_3(o))
                o = o * target_ratio
                output = o + target_ratio
                return output

        class Ratio_Base_Modified_Mode(tor.nn.Module):
            def __init__(self):
                super(Ratio_Base_Modified_Mode, self).__init__()
                # input shape: (48) = (14, 5. 14, 14, 1) = (sample, userinfo, target_sample_avg, nontarget_sample_avg, target_ratio)
                self.fc_1 = tor.nn.Linear(5 + 14 + 14, 2 ** 10)
                self.fc_2 = tor.nn.Linear(2 ** 10, 2 ** 10)
                self.fc_3 = tor.nn.Linear(2 ** 10, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.tanh = tor.nn.Tanh()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                sample, userinfo, target_sample_avg, nontarget_sample, target_ratio = x[:, 0:14], x[:, 14:19], x[:, 19:33], x[:, 33:47], x[:, 47:48]
                x = tor.cat((userinfo, tor.abs(sample - target_sample_avg), tor.abs(sample - nontarget_sample)), 1)
                o = self.fc_1(x)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.tanh(self.fc_3(o))
                norm = tor.min(tor.cat(((1 - target_ratio), target_ratio), 1), dim=1)[0].reshape(-1, 1)
                o = o.mul(norm)
                output = o + target_ratio
                return output
                
        if model_name == 'nn':
            model = NN()
        elif model_name == 'ratio_base':
            model = Ratio_Base_Mode()
        elif model_name == 'ratio_base_modified':
            model = Ratio_Base_Modified_Mode()

        return model
    
    
    def train(self):
        ### Load date
        data_raw = np.load(self.data_fp)
        x_train = data_raw[:, :-1]
        y_train = data_raw[:, -1]
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        print('|Show training samples', flush=True)
        print('|Train data size: {}'.format(x_train.shape))
        print('|Ratio of 1 in training data: {}'.format(sum(y_train == 1) / y_train.shape[0]))
        print(x_train[:2])
        print(y_train[:30])

        if self.gpu:
            tor.cuda.set_device(0)
        model = self.create_model('ratio_base_modified').cuda() if self.gpu else self.create_model('ratio_base_modified')
        loss_func = tor.nn.MSELoss().cuda() if self.gpu else tor.nn.MSELoss()
        optim = tor.optim.Adam(model.parameters(), lr=self.lr)


        ### Training
        print('|Params: ')
        print('|Use GPU: {}'.format(self.gpu))
        print('|lr:{} batch:{} start_time: {}'.format(self.lr, self.batch_size, self.start_time), flush=True)

        for epoch in range(30):
            print('|Epoch: ', epoch + 1, flush=True)
            total_loss = 0
            for step, (x, y) in enumerate(data_loader):
                x = Variable(x) if not self.gpu else Variable(x).cuda()
                y = Variable(y) if not self.gpu else Variable(y).cuda()
                optim.zero_grad()
                pred = model(x)
                
                loss = loss_func(pred, y)
                total_loss += loss
                loss.backward()
                optim.step()

                if step % 2000 == 0:
                    print('|Loss: ', float(total_loss / 2000), flush=True)
                    total_loss = 0

                if step % 10000 == 0:
                    model.eval()
                    chosen_index_list = np.random.choice(range(len(x_train)), 10 ** 4, replace=False)
                    x_valid = tor.FloatTensor(x_train[chosen_index_list])
                    x_valid = Variable(x_valid).cuda() if self.gpu else x_valid
                    y_valid = tor.FloatTensor(y_train[chosen_index_list])
                    pred = model(x_valid).clone().cpu().detach().view(-1)
                    acc = tor.mean(tor.tensor((pred > 0.5) == (y_valid == 1), dtype=tor.float), dim=0)
                    print('|Acc: ', float(acc.detach()))
                    model.train()

            tor.save(model.state_dict(), './models/model_me_{}.pkl'.format(self.start_time))



class Modified_Trainer():
    def __init__(self, argv):
        self.data_fp = argv[0]
        self.gpu = bool(int(argv[1]))
        self.lr = 10 ** -4
        self.batch_size = 2 ** 3
        self.start_time = int(time.time())


    @staticmethod
    def create_model(model_name='nn'):
        class NN(tor.nn.Module):
            def __init__(self):
                super(NN, self).__init__()
                # input shape: (10)
                self.fc_1 = tor.nn.Linear(10, 2 ** 10)
                self.fc_2 = tor.nn.Linear(2 ** 10, 2 ** 10)
                self.fc_3 = tor.nn.Linear(2 ** 10, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                o = self.fc_1(x)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.fc_3(o)
                output = self.sig(o)
                return output

        if model_name == 'nn':
            model = NN()
        
        return model


    def train(self):
        ### Load date
        data_raw = np.load(self.data_fp)
        x_train = data_raw[:, :-1]
        y_train = data_raw[:, -1]
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        print('|Show training samples', flush=True)
        print('|Train data size: {}'.format(x_train.shape))
        print('|Ratio of 1 in training data: {}'.format(sum(y_train == 1) / y_train.shape[0]))
        print(x_train[:2])
        print(y_train[:30])

        if self.gpu:
            tor.cuda.set_device(0)
        model = self.create_model('nn').cuda() if self.gpu else self.create_model('nn')
        loss_func = tor.nn.MSELoss().cuda() if self.gpu else tor.nn.MSELoss()
        optim = tor.optim.Adam(model.parameters(), lr=self.lr)


        ### Training
        print('|Params: ')
        print('|Use GPU: {}'.format(self.gpu))
        print('|lr:{} batch:{} start_time: {}'.format(self.lr, self.batch_size, self.start_time), flush=True)

        for epoch in range(30):
            print('|Epoch: ', epoch + 1, flush=True)
            total_loss = 0
            for step, (x, y) in enumerate(data_loader):
                x = Variable(x) if not self.gpu else Variable(x).cuda()
                y = Variable(y) if not self.gpu else Variable(y).cuda()
                optim.zero_grad()
                pred = model(x)
                loss = loss_func(pred, y)
                total_loss += loss
                loss.backward()
                optim.step()

                if step % 2000 == 0:
                    print('|Loss: ', float(total_loss / 2000), flush=True)
                    total_loss = 0

                if step % 10000 == 0:
                    model.eval()
                    chosen_index_list = np.random.choice(range(len(x_train)), 10 ** 4, replace=False)
                    x_valid = tor.FloatTensor(x_train[chosen_index_list])
                    x_valid = Variable(x_valid).cuda() if self.gpu else x_valid
                    y_valid = tor.FloatTensor(y_train[chosen_index_list])
                    pred = model(x_valid).clone().cpu().detach().view(-1)
                    acc = tor.mean(tor.tensor((pred > 0.5) == (y_valid == 1), dtype=tor.float), dim=0)
                    print('Acc: ', float(acc.detach()))
                    model.train()

            tor.save(model.state_dict(), './models/model_me_{}.pkl'.format(self.start_time))



class DT_Trainer():
    def __init__(self, argv):
        self.data_fp = argv[0]
        self.data_test_fp = argv[1]


    def train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, auc, roc_curve
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier

        ### Load date
        data_raw = np.load(self.data_fp)
        data_raw = data_raw.astype(np.float)
        data_test = np.load(self.data_test_fp)

        x_data = data_raw[:, :-1]
        y_data = data_raw[:, -1]
        x_train_pair, x_valid_pair, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
        x_train, x_train_stat, x_valid, x_valid_stat = x_train_pair, x_train_pair[:, -1], x_valid_pair, x_valid_pair[:, -1]
        x_test = data_test[:, :-1]
        x_test_stat = data_test[:, -2].reshape(-1)
        
        print('|Train data size: {}'.format(x_train.shape))
        print('|Ratio of 1 in training data: {}'.format(sum(y_valid == 1) / y_valid.shape[0]))


        ### Training
        for depth in [3, 5, 10]:
            model = XGBClassifier(
                booster='gbtree',
                max_depth=depth,
                n_estimators=50,
                subsample=0.5,
                learning_rate=0.01,
                colsample_bytree=0.5,
                colsample_bylevel=0.8,
                eval_metric='auc',
                verbosity=2,
                n_jobs=2,
            )
            model.fit(x_train, y_train)
            score = model.score(x_valid, y_valid)
            print('|Score on validation set: ', score)

            pred_valid = model.predict_proba(x_valid)[:, 1]
            score_best, alpha_best = 0, 0
            for alpha in np.arange(0, 1.1, 0.1):
                alpha = round(alpha, 2)
                pred_comb = pred_valid * alpha + x_valid_stat * (1 - alpha)
                pred_label = [ 1 if p >= 0.5 else 0 for p in pred_comb]
                fpr, tpr, thresholds = roc_curve(y_valid, pred_comb)
                # score = accuracy_score(y_valid, pred_label)
                auc_ = auc(fpr, tpr)
                score = auc_
                print('|auc:', auc_)
                if score > score_best:
                    score_best, alpha_best = score, alpha
            print('|Best alpha:', alpha_best)
            print('|Best score:', score_best)

            pred_xgboost = model.predict_proba(x_test)[:, 1]
            pred_avg_array = pred_xgboost * alpha_best + x_test_stat * (1 - alpha_best)

            pred_fp = './predictions/prediction_{}_{}.csv'.format(mode, int(time.time()))
            print('|Prediction file: ', pred_fp)
            pd.DataFrame(pred_avg_array).to_csv(pred_fp, index_label='id', header=['target'])



class XGBoost_Trainer():
    def __init__(self, argv):
        self.data_fp = argv[0]
        self.data_test_fp = argv[1]


    def train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, auc, roc_curve
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier, plot_tree

        ### Load date
        data_raw = np.delete(np.load(self.data_fp), 1, axis=1)
        data_test = np.load(self.data_test_fp)[:, 1:].astype(np.float)
        data_test = np.delete(np.load(self.data_test_fp), 1, axis=1)

        x_train, x_valid, y_train, y_valid = train_test_split(data_array[:, :-1], data_array[:, -1], test_size=0.3, random_state=0)

        x_test = data_test[:, :-1]

        print('|Show training samples', flush=True)
        print('|Train data size: {}'.format(x_train.shape))
        print('|Ratio of 1 in training data: {}'.format(sum(y_valid == 1) / y_valid.shape[0]))


        ### Training
        for depth, min_child_weight, n_estimators in product([30, 50, 100, 150], [1, 5, 10, 50, 100,], [50, 100]):
            print('\n|depth: {} |min_child_weight: {} |n_estimators: {}'.format(depth, min_child_weight, n_estimators))
            model = XGBClassifier(
                booster='gbtree',
                max_depth=depth,
                n_estimators=n_estimators,
                min_child_weight=min_child_weight,
                subsample=0.5,
                learning_rate=0.1,
                colsample_bytree=0.5,
                colsample_bylevel=0.8,
                # eval_metric='auc',
                n_jobs=4,
            )

            model.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric='auc',
                early_stopping_rounds=6,
                verbose=True
            )

            evals_result = model.evals_result()
            print('|Feature importances: ', model.feature_importances_)
            
            pred_valid = model.predict_proba(x_valid)[:, 1].reshape(-1)
            fpr, tpr, thresholds = roc_curve(y_valid, pred_valid)
            auc_ = auc(fpr, tpr)
            score = auc_
            print('|auc:', auc_, flush=True)
            
            pred_xgboost = model.predict_proba(x_test)[:, 1]
            pred_avg_array = pred_xgboost
            

            ### Output the prediction file
            pred_fp = './predictions/prediction_{}_{}.csv'.format(mode, int(time.time()))
            pd.DataFrame(pred_avg_array).to_csv(pred_fp, index_label='id', header=['target'])

            print('|Prediction file: ', pred_fp, flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    # sk_nn(data_fp, target_fp)
    # nn(data_fp, target_fp)
    argv = sys.argv[2:] if len(sys.argv) > 2 else None

    if mode == 'memory':
        trainer = Memory_Based_Trainer(argv)
        # argv: data_fp, isGpu
    elif mode == 'modified':
        # argv: data_fp, isGpu
        trainer = Modified_Trainer(argv)
        # argv: data_fp
    elif mode == 'dt':
        # argv: data_fp, data_test_fp
        trainer = DT_Trainer(argv)
    elif mode == 'xg':
        # argv: data_fp, data_test_fp
        trainer = XGBoost_Trainer(argv)
    
    trainer.train()