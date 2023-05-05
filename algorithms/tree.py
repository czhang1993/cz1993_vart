import numpy as np


def entropy(p):
    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def info_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    ig_p = entropy(p_parent)
    ig_l = entropy(p_left)
    ig_r = entropy(p_right)
    out = ig_p
    out -= len(left_child) / len(parent) * ig_l
    out -= len(right_child) / len(parent) * ig_r
    return out


def draw_boot(x_train, y_train):
    boot_indices = list(
        np.random.choice(
            range(len(x_train)), 
            len(x_train), 
            replace = True
        )
    )
    oob_indices = [i for i in range(len(x_train)) if i not in boot_indices]
    x_boot = x_train.iloc[boot_indices].values
    y_boot = y_train[boot_indices]
    x_oob = x_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]
    return x_boot, y_boot, x_oob, y_oob


def oob_score(tree, x_test, y_test):
    mis_label = 0
    for i in range(len(x_test)):
        pred = predict_tree(tree, x_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(x_test)
