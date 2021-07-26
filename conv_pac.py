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
class Conv():
    def __init__(self,input_shape, output_shape, max_move = .5):
        self.max_move = .5
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv_shape = list(self.input_shape)
        # print(self.conv_shape)
        for i in range(-1, -len(self.output_shape) - 1, -1):
            self.conv_shape[i] -= self.output_shape[i]
            # print(self.conv_shape)

        # print(self.conv_shape)
        self.values = np.random.random_sample(tuple(self.conv_shape))

    def transform(self, input):
        assert input.shape == self.input_shape
        output = np.zeros(self.output_shape)

        for a in range(len(self.input_shape)):




class Conv_PAC():

    def __init__(self, cap_loss = .65, decay = .999, decay_increase = .0001, loss_decay = False):
        self.max_loss = cap_loss
        self.max_decay = decay
        self.decay_reset = .9
        self.decay = decay
        self.loss_decay = loss_decay
        self.decay_speed = decay_increase

    def fit(self, x_train, y_train, restrict = None):
        # print(f'restrict is {restrict}')
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

            # # restrict the possible predictions
            # if restrict:
            #     # print(self.classes, self.class_rev)
            #
            #     for i in range(len(predict)):
            #         if i not in non_zeroed:
            #             predict[i] = np.NINF


            if not max(predict) != predict[self.classes[y_train[index]]-1]:
                error = max(predict) - 1
                loss = max(0, 1-error)
                if self.loss_decay:
                    loss *= self.decay
                    self.decay *= (1-self.decay_speed)
                    if self.decay <= .2:
                        self.decay = self.max_decay * self.decay_reset
                        self.decay_reset *= self.decay_reset
                elif self.max_loss:
                    loss = min(self.max_loss, loss)
                # τ = error/(2*np.linalg.norm(data_norm))
                # τ = 1
                # print(τ)
                if restrict:
                    non_zeroed = [self.class_rev[a] for a in restrict[index]]
                    y[np.argmax(predict)] =  -1 if np.argmax(predict) in non_zeroed else -2
                else:
                    y[np.argmax(predict)] = -1
                # print(data_norm.shape, (y.T*loss).shape)
                # print(data_norm.ndim, (y.T*loss).ndim)
                self.weights = self.weights + y.T * loss *data_norm #* τ

    def predict(self, x_test, restrict = None):
        x_test = np.array([b.flatten() for b in x_test])
        predictions = []
        for index, data_point in enumerate(x_test):
            data_norm = np.array(data_point).reshape((data_point.shape[0],1))
            # print(data_norm* self.weights)
            predict = np.dot(self.weights.T, data_norm)
            if restrict:
                non_zeroed = [self.class_rev[a] for a in restrict[index]]
                for i in range(len(predict)):
                    if i not in non_zeroed:
                        predict[i] = np.NINF
            predictions.append(np.argmax(predict))

        # predictions = [x if x != 0 else 1 for x in predictions]
        predictions = np.array(predictions)
        # print(predictions)

        return [self.class_rev[int(pred)] for pred in predictions]

if __name__ == "__main__":
    a = Conv((7,5,5), (2,2))
