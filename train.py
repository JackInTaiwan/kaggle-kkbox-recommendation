import sys
import time
import pandas as pd
import torch as tor
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



def sk_nn(data_fp, target_fp):
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier as Classifier


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
        hidden_layer_sizes=2 ** 11, activation='relu',
        solver='sgd', batch_size=2 ** 3,
        learning_rate_init=10 ** -3,
        tol=1e-6, n_iter_no_change=30,
        max_iter=200, shuffle=True,
        verbose=True
    )

    classifier.fit(x_train, y_train)

    pred = classifier.predict(x_train)
    score = classifier.score(x_train, y_train)
    print(score)

    joblib.dump(classifier, './models/model_{}.pkl'.format(time.time()))



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



def nn(data_fp, target_fp):
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
    nn = NNModel()
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
                print(float(total_loss / 1000))
                total_loss = 0

            if step % 10000 == 0:
                nn.eval()
                pred = nn(tor.FloatTensor(x_train)).detach()
                y_train = tor.FloatTensor(y_train)
                acc = tor.mean(tor.tensor((pred > 0.5) == (y_train == 1), dtype=tor.float), dim=0)
                print('Acc: ', float(acc))
                nn.train()

        ### Validation loss
        optim.zero_grad()
        tor.save(nn.state_dict(), './models/model_nn_3_{}.pkl'.format(int(time.time())))



class Memory_Based_Trainer():
    def __init__(self, argv):
        self.data_fp = argv[0]
        self.lr = 10 ** -3
        self.batch_size = 2 ** 3
        self.start_time = int(time.time())

    def create_model(self):
        class NN(tor.nn.Module):
            def __init__(self):
                super(NN, self).__init__()
                # input shape: (48) = (14, 5. 14, 14, 1) = (sample, userinfo, target_sample_avg, nontarget_sample_avg, target_ratio)
                self.fc_1 = tor.nn.Linear(5 + 14 + 14 + 1, 2 ** 9)
                self.fc_2 = tor.nn.Linear(2 ** 9, 2 ** 9)
                self.fc_3 = tor.nn.Linear(2 ** 9, 1)
                self.relu = tor.nn.ReLU()
                self.sig = tor.nn.Sigmoid()
                self.drop = tor.nn.Dropout(p=0.5)

            def forward(self, x) :
                sample, userinfo, target_sample_avg, nontarget_sample, target_ratio = x[:, 0:14], x[:, 14:19], x[:, 19:33], x[:, 33:47], x[:, 47:48]
                o = tor.cat((userinfo, (sample - target_sample_avg), (sample - nontarget_sample), target_ratio), 1)
                o = self.fc_1(o)
                o = self.drop(self.relu(o))
                o = self.fc_2(o)
                o = self.drop(self.relu(o))
                o = self.fc_3(o)
                output = self.sig(o) 

                return output
        model = NN()
        return model
    
    
    def train(self):
        ### Load date
        data_row = np.load(self.data_fp)
        x_train = data_row[:, :-1]
        y_train = data_row[:, -1]
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=2 ** 3,
            shuffle=True,
            drop_last=True,
        )

        print('|Show training samples')
        print('|Train data size: {}'.format(x_train.shape))
        print('|Ratio of 1 in training data: {}'.format(sum(y_train == 1) / y_train.shape[0]))
        print(x_train[:5])
        print(y_train[:30])

        model = self.create_model()
        loss_func = tor.nn.MSELoss()
        optim = tor.optim.Adam(model.parameters(), lr=self.lr)


        ### Training
        print('|Params: ')
        print('|lr:{} batch:{} start_time: {}'.format(self.lr, self.batch_size, self.start_time))

        for epoch in range(30):
            print('|Epoch: ', epoch + 1)
            total_loss = 0
            for step, (x, y) in enumerate(data_loader):
                optim.zero_grad()
                pred = model(x)
                
                loss = loss_func(pred, y)
                total_loss += loss
                loss.backward()
                optim.step()

                if step % 2000 == 0:
                    print(float(total_loss / 2000))
                    total_loss = 0

                if step % 10000 == 0:
                    model.eval()
                    chosen_index_list = np.random.choice(range(len(x_train)), 10 ** 4, replace=False)
                    x_valid = tor.FloatTensor(x_train[chosen_index_list])
                    y_valid = tor.FloatTensor(y_train[chosen_index_list])
                    pred = model(x_valid).clone().detach().view(-1)

                    acc = tor.mean(tor.tensor((pred > 0.5) == (y_valid == 1), dtype=tor.float), dim=0)
                    print('Acc: ', float(acc.detach()))
                    print(pred.detach().numpy())
                    model.train()

            tor.save(nn.state_dict(), './models/model_me_{}.pkl'.format(self.start_time))



if __name__ == '__main__':
    mode = sys.argv[1]
    # sk_nn(data_fp, target_fp)
    # nn(data_fp, target_fp)
    argv = sys.argv[2:] if len(sys.argv) > 2 else None

    if mode == 'memory':
        trainer = Memory_Based_Trainer(argv)
        trainer.train()