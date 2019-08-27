import numpy as np

import tcn


'''
data = np.load('actions.npz')
keypoints = data['keypoints']
actions = data['actions']
print(actions.shape)
lenght = actions[..., 1] - actions[..., 0]
actions = actions[np.logical_and(4 < lenght, lenght < 64)]
print(actions.shape)
n = keypoints.shape[0]
y = np.zeros(n, dtype='int32')
w = np.zeros(n, dtype='float32')
for i, (start, last, action) in enumerate(actions):
    for j in range(start, last + 1):
        y[j] = action
        w[j] = (j - start + 1) / (last - start + 1)
print(y.shape, w.shape)
np.savez('actions.v2.npz', x=keypoints, y=y, w=w, labels=data['labels'])
'''
data = np.load('actions.v2.npz')
x = data['x'][..., :2]
y = data['y']
w = data['w']
l = data['labels']

print(x.shape)
print(y.shape)
print(w.shape)
x -= x.mean(1).reshape(-1, 1, 2)
x /= (x.max(1) - x.min(1) + 0.000001).reshape(-1, 1, 2)
i = 1404
print(x[i])
print('y =', l[y[i]], ';w =', w[i])
x = x.reshape((1, -1, 12))
y.shape = 1, -1, 1
w.shape = 1, -1

from keras.models import Input, Model
from keras.layers import Dense, Activation, BatchNormalization
i = Input(batch_shape=(None, x.shape[1], 12))
o = tcn.TCN(return_sequences=True)(i)
o = Dense(y.max()+1)(o)
o = Activation('softmax')(o)
m = Model(i, o)
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  sample_weight_mode="temporal")
m.fit(x, y, sample_weight=w, epochs=10)
m.save('first.h5', include_optimizer=False)
