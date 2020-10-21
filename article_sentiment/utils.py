import torch


def num_correct(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    num_correct = (max_indices == Y).sum().data.cpu().numpy()
    return num_correct


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc