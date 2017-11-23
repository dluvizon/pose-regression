# -*- coding: utf-8 -*-
import numpy as np


def norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2), axis=axis))

def valid_joints(y, min_valid=-1e6):
    def and_all(x):
        if x.all():
            return 1
        return 0

    return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))


def pckh(y_true, y_pred, head_size, refp=0.5):
    """Compute the PCKh measure (using refp of the head size) on predicted
    samples, considering the PA16J pose layout (see file pose.py).

    # Arguments
        y_true: [num_samples, nb_joints, 2]
        y_pred: [num_samples, nb_joints, 2]
        head_size: [num_samples, 1]

    # Return
        The PCKh score.
    """

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(head_size)
    num_samples = len(y_true)

    # Ignore the joints pelvis and thorax (respectively 0 and 1 on the PA16J
    # pose layout.
    used_joints = range(2, 16)
    y_true = y_true[:, used_joints, :]
    y_pred = y_pred[:, used_joints, :]
    dist = np.zeros((num_samples, len(used_joints)))
    valid = np.zeros((num_samples, len(used_joints)))

    for i in range(num_samples):
        valid[i,:] = valid_joints(y_true[i])
        dist[i,:] = norm(y_true[i] - y_pred[i], axis=1) / head_size[i]
    match = (dist <= refp) * valid

    return match.sum() / valid.sum()

