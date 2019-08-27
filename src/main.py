import numpy as np
import tcn
from keras import models
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Activation


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

    return x, y, w


def build_model(x, y):
    i = models.Input(batch_shape=(None, x.shape[1], x.shape[2]))
    o = tcn.TCN(return_sequences=True)(i)
    o = Dense(y.max() + 1)(o)
    o = Activation('softmax')(o)
    m = models.Model(i, o)
    return m


data = np.load('actions.npz')
l = data['labels']
x, y, w = preproccess(data['keypoints'], data['actions'])
print(x.shape, y.shape, w.shape)
x.shape = 1, -1, 12
y.shape = 1, -1, 1
w.shape = 1, -1

# m = build_model(x, y)
m = models.load_model('first.h5')
# 当标准评估停止提升时，降低学习速率
reduce_lr = ReduceLROnPlateau(verbose=1)
m.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          sample_weight_mode='temporal')
m.fit(x, y, sample_weight=w, epochs=1600, callbacks=[reduce_lr])
# m.save('first.h5', include_optimizer=False)
m.save('first.h5')
