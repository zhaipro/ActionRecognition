import time

import numpy as np
from keras import models
from keras.layers import Dense, Activation
from keras.utils import Sequence

import tcn


def preproccess(keypoints, actions):
    # 动作的结束点减起始点，从而获取每一个动作的长度
    lenght = actions[..., 1] - actions[..., 0]
    # 过滤掉太长的和太短的，甚至长度为负数的动作…
    actions = actions[np.logical_and(4 < lenght, lenght < 64)]
    # 将其翻译成方便机器学习的数据结构
    n = keypoints.shape[0]
    y = np.zeros(n, dtype='int32')
    w = np.zeros(n, dtype='float32')
    for i, (start, last, action) in enumerate(actions):
        for j in range(start, last + 1):
            y[j] = action
            w[j] = (j - start + 1) / (last - start + 1)

    # 开始进行真正的预处理
    epsilon = 1e-7
    # 忽略置信度
    x = keypoints[..., :2].copy()
    x -= x.mean(1, keepdims=True)
    x /= x.max(1, keepdims=True) - x.min(1, keepdims=True) + epsilon

    # 定义：自然放下的结束点是可以切割的点，从而方便对动作进行洗牌
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html
    cut = actions[actions[:, 2] == 2][:, 1]
    x = np.split(x, cut)
    y = np.split(y, cut)
    w = np.split(w, cut)

    return x, y, w


# https://keras.io/zh/utils/#sequence
# https://github.com/keras-team/keras/issues/9707
class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x, y, w):
        self.x, self.y = np.array(x), np.array(y)
        self.w = np.array(w)
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = np.concatenate(self.x[self.indices])
        y = np.concatenate(self.y[self.indices])
        w = np.concatenate(self.w[self.indices])
        x.shape = 1, -1, 12
        y.shape = 1, -1, 1
        w.shape = 1, -1
        return x, y, w

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def build_model(x, y):
    i = models.Input(batch_shape=(None, None, x.shape[2]))
    o = tcn.TCN(nb_filters=32, dropout_rate=0.15, return_sequences=True)(i)
    o = Dense(y.max() + 1)(o)
    o = Activation('softmax')(o)
    m = models.Model(i, o)
    return m


def _load_data(data):
    x, y, w = preproccess(data['keypoints'], data['actions'])
    n = int(len(x) * 0.9)
    x_train = x[:n]
    y_train = y[:n]
    w_train = w[:n]

    x_test = np.concatenate(x[n:])
    y_test = np.concatenate(y[n:])
    w_test = np.concatenate(w[n:])
    x_test.shape = 1, -1, 12
    y_test.shape = 1, -1, 1
    w_test.shape = 1, -1

    return (x_train, y_train, w_train), (x_test, y_test, w_test)


def evaluate():
    data = np.load('actions.npz')
    _, (x_test, y_test, w_test) = _load_data(data)
    m = models.load_model('model.1567145427.h5')
    m.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              sample_weight_mode='temporal')
    r = m.evaluate(x_test, y_test, sample_weight=w_test)
    print(r)


if __name__ == '__main__':
    # evaluate()
    # exit()
    data = np.load('actions.npz')
    (x, y, w), (x_test, y_test, w_test) = _load_data(data)
    m = build_model(x_test, y_test)
    m.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              sample_weight_mode='temporal')

    train_datagen = Generator(x, y, w)
    m.fit_generator(train_datagen, epochs=1600,
                    steps_per_epoch=1,
                    validation_data=(x_test, y_test, w_test))

    m.save(f'model.{int(time.time())}.h5', include_optimizer=False)
