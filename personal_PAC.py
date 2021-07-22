# custom PAC

# following from https://www.youtube.com/watch?v=TJU8NfDdqNQ
# Text Classification 3: Passive Aggressive Algorithm

import numpy as np
from sklearn.preprocessing import normalize
'''
1. Initialize weights to zeros
2. Get the input document
3. Normalize the data (d)
4. Predict positive if d_transp X w > 0 (d_transp X w -> info)
5. observe true class (y +-1)
6. want y(info) >= 1
7. make loss: max(0, 1-y(info)) (L)
8. update w_n = w + y*L*d

'''
class PAC():

    def __init__(self, cap_loss = .65, decay = .001, decay_increase = .0001, loss_decay = False):
        self.max_loss = cap_loss
        self.decay = .001
        self.loss_decay = loss_decay
        self.decay_speed = decay_increase

    def fit(self, x_train, y_train):
        # flatten x_train
        x_train = np.array([b.flatten() for b in x_train])
        # get classes
        # this part will need adjusting if there are more than two classes
        y_train = np.array(y_train)
        self.classes = {}
        classes = set(y_train)
        # assert len(classes) == 2, "more than two classes"
        val = 0
        for c in classes:
            self.classes[c] = val
            val += 1
        self.class_rev = dict([reversed(i) for i in self.classes.items()])

        #initialize weights to 0
        print(x_train.shape)
        print(y_train.shape)
        self.weights = np.zeros((x_train.shape[1], len(self.classes)))

        for index, data_point in enumerate(x_train):
            # because all 0's and 1's no need to normalize
            data_norm = np.array(data_point).reshape((data_point.shape[0],1))

            predict = np.dot(self.weights.T, data_norm)
            # print(predict)
            y = np.zeros((predict.shape[0], 1))
            y[self.classes[y_train[index]]-1] = 1
            # print(predict)
            if not max(predict) != predict[self.classes[y_train[index]]-1]:
                error = max(predict) - 1
                loss = max(0, 1-error)
                if self.loss_decay:
                    loss *= (1-self.decay)
                    self.decay += self.decay_speed
                else:
                    loss = min(self.max_loss, loss)
                τ = error/(2*np.linalg.norm(data_norm))[0]
                print(τ)
                y[np.argmax(predict)] =  -1
                # print(data_norm.shape, (y.T*loss).shape)
                # print(data_norm.ndim, (y.T*loss).ndim)
                self.weights = self.weights + y.T * loss *data_norm * τ

    def predict(self, x_test):
        x_test = np.array([b.flatten() for b in x_test])
        predictions = []
        for data_point in x_test:
            data_norm = np.array(data_point).reshape((data_point.shape[0],1))
            # print(data_norm* self.weights)
            predictions.append(np.argmax(np.dot(self.weights.T, data_norm)))

        # predictions = [x if x != 0 else 1 for x in predictions]
        predictions = np.array(predictions)
        # print(predictions)

        return [self.class_rev[int(pred)] for pred in predictions]
