import numpy as np

# import tcn


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
