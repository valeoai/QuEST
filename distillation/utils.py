from __future__ import print_function

import pickle
import numpy as np
import torch
import torch.nn as nn


def load_pickle_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        if isinstance(data, dict):
            data = {k.decode('ascii'): v for k, v in data.items()}

    return data


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class BaseMeter:
    def __init__(self, name=None, _run=None):
        """ If singleton is True, work with single values."""
        self.val = None  # saves the last value added
        self.sum = None
        self.count = 0
        self.name = name
        self._run = _run

    def reset(self):
        self.val = None
        self.sum = None
        self.count = 0

    def update(self, array):
        array = np.array(array)
        self.val = array.astype(np.float64)
        if self.count == 0:
            self.sum = np.copy(self.val)
        else:
            self.sum += self.val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count

    def log(self):
        if self.name is None:
            return
        # we can't log metrics in 2D.
        if self.sum.ndim > 1:
            return

        if self._run is None:
            return

        average = self.sum / self.count
        if average.ndim == 0:
            self._run.log_scalar(self.name, average)
        else:
            for i, scalar in enumerate(average):
                self._run.log_scalar(f'{self.name}_{i}', scalar)

    def __str__(self):
        average = self.sum / self.count
        if average.ndim == 0:
            return f'{average:.4f}'
        elif average.ndim == 1:
            return ' '.join(f'{scalar:.4f}' for scalar in average)
        elif average.ndim >1:
            raise TypeError('cannot print easily a numpy array'
                            'with a dimension above 1')


class DAverageMeter:
    def __init__(self, name=None, _run=None):
        self.values = {}
        self.name = name
        self._run = _run

    def reset(self):
        self.values = {}

    def update(self, values):
        assert(isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int, list)):
                if not (key in self.values):
                    self.values[key] = BaseMeter(self.make_name(key), self._run)
                self.values[key].update(val)
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter(self.make_name(key), self._run)
                self.values[key].update(val)
            else:
                raise TypeError(f'Wrong type {type(val)}')

    def average(self):
        return {k: _get_average(v) for k, v in self.values.items()}

    def __str__(self):
        return str({k: str(v) for k, v in self.values.items()})

    def make_name(self, key):
        return f'{self.name}_{key}'

    def log(self):
        for value in self.values.values():
            value.log()


def _get_average(value):
    if isinstance(value, DAverageMeter):
        return value.average()
    return value.avg


def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float).mean() * 100, top5_correct.astype(float).mean() * 100


def top1accuracy(output, target):
    return top1accuracy_tensor(output, target).item()


def top1accuracy_tensor(output, target):
    pred = output.max(dim=1)[1]
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


def add_dimension(tensor, dim_size):
    assert((tensor.size(0) % dim_size) == 0)
    return tensor.view(
        [dim_size, tensor.size(0) // dim_size,] + list(tensor.size()[1:]))
