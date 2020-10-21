import torch
# import numpy as np
# from itertools import product
# import matplotlib.pyplot as plt


def num_correct(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    num_correct = (max_indices == Y).sum().data.cpu().numpy()
    return num_correct


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


# def plot_cm(cm, labels):
#     assert cm.ndim == 2
#     m, n = cm.shape
#     assert m == n
#     assert m == len(labels)
#
#     fig, ax = plt.subplots()
#     im_ = ax.imshow(cm, interpolation='nearest', cmap='viridis')
#     cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
#     text_ = np.empty_like(cm, dtype=object)
#
#     thresh = (cm.max() + cm.min()) / 2.0
#     for i, j in product(range(n), range(n)):
#         color = cmap_max if cm[i, j] < thresh else cmap_min
#         text_cm = format(cm[i, j], 'd')
#         text_[i, j] = ax.text(
#             j, i, text_cm,
#             ha="center", va="center",
#             color=color)
#     fig.colorbar(im_, ax=ax)
#     ax.set_x
