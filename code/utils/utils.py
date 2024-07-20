import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def weighted_binary_cross_entropy(target, epsilon=1e-6):
    """
    """
    with torch.no_grad():
        num_class_0 = (target == 0).sum().float()
        num_class_1 = (target == 1).sum().float()

        weight_for_class_0 = num_class_1 / (num_class_0 + epsilon)
        weight_for_class_1 = torch.tensor(1.0).to(target.device)

        weights = [weight_for_class_0, weight_for_class_1]

    return weights

def class_wise_accuracy(output, target):
    """
    Calculate class-wise accuracy for binary classification.

    Parameters:
    - output: the probabilities as predicted by the model.
    - target: the binary ground truth labels.

    Returns:
    - A tuple of accuracies for class 0 and class 1.
    """
    predicted_classes = (output > 0.5).float()
    correct = (predicted_classes == target).float()
    class_0_accuracy = (correct * (target == 0)).sum() / ((target == 0).sum().float())
    class_1_accuracy = (correct * (target == 1)).sum() / ((target == 1).sum().float())
    return class_0_accuracy.item(), class_1_accuracy.item()

import itertools
from torch.utils.data.sampler import Sampler
def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


