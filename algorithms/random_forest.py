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


def find_split_point(
    x_boot,
    y_boot,
    max_features
):
    feature_ls = list()
    num_features = len(x_boot[0])

    while len(feature_ls) <= max_features:
    feature_idx = random.sample(range(num_features), 1)
    if feature_idx not in feature_ls:
        feature_ls.extend(feature_idx)

    best_info_gain = -999
    node = None
    for feature_idx in feature_ls:
    for split_point in x_boot[:,feature_idx]:
        left_child = {"x_boot": [], "y_boot": []}
        right_child = {"x_boot": [], "y_boot": []}

        # split children for continuous variables
        if type(split_point) in [int, float]:
            for i, value in enumerate(x_boot[:,feature_idx]):
                if value <= split_point:
                    left_child["x_boot"].append(x_boot[i])
                    left_child["y_boot"].append(y_boot[i])
                else:
                    right_child["x_boot"].append(x_boot[i])
                    right_child["y_boot"].append(y_boot[i])
        # split children for categoric variables
        else:
            for i, value in enumerate(x_boot[:, feature_idx]):
                if value == split_point:
                    left_child["x_boot"].append(x_boot[i])
                    left_child["y_boot"].append(y_boot[i])
                else:
                    right_child["x_boot"].append(x_boot[i])
                    right_child["y_boot"].append(y_boot[i])

        split_info_gain = info_gain(
            left_child["y_boot"], 
            right_child["y_boot"]
        )
        if split_info_gain > best_info_gain:
            best_info_gain = split_info_gain
            left_child["x_boot"] = np.array(left_child["x_boot"])
            right_child["x_boot"] = np.array(right_child["x_boot"])
            node = {
                "info_gain": split_info_gain,
                "left_child": left_child,
                "right_child": right_child,
                "split_point": split_point,
                "feature_idx": feature_idx
            }
    return node


def terminal_node(node):
    y_boot = node["y_boot"]
    pred = max(y_boot, key = y_boot.count)
    return pred


def split_node(
    node,
    max_features,
    min_samples_split,
    max_depth, depth
):
    left_child = node["left_child"]
    right_child = node["right_child"]

    del(node["left_child"])
    del(node["right_child"])

    if len(left_child["y_boot"]) == 0 or len(right_child["y_boot"]) == 0:
        empty_child = {
            "y_boot": left_child["y_boot"] + right_child["y_boot"]
        }
        node["left_split"] = terminal_node(empty_child)
        node["right_split"] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node["left_split"] = terminal_node(left_child)
        node["right_split"] = terminal_node(right_child)
        return node

    if len(left_child["x_boot"]) <= min_samples_split:
        node["left_split"] = node["right_split"] = terminal_node(left_child)
    else:
        node["left_split"] = find_split_point(
            left_child["x_boot"],
            left_child["y_boot"],
            max_features
        )
        split_node(
            node["left_split"],
            max_depth,
            min_samples_split,
            max_depth,
            depth + 1
        )
    if len(right_child["x_boot"]) <= min_samples_split:
        node["right_split"] = node["left_split"] = terminal_node(right_child)
    else:
        node["right_split"] = find_split_point(
            right_child["x_boot"],
            right_child["y_boot"],
            max_features
        )
        split_node(
            node["right_split"],
            max_features,
            min_samples_split,
            max_depth,
            depth + 1
        )
        
        
def build_tree(
    x_boot,
    y_boot,
    max_depth,
    min_samples_split,
    max_features
):
    root_node = find_split_point(
        x_boot,
        y_boot,
        max_features
    )
    split_node(
        root_node,
        max_features,
        min_samples_split,
        max_depth,
        1
    )
    return root_node


def random_forest(
    x_train,
    y_train,
    n_estimators,
    max_features,
    max_depth,
    min_samples_split
):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        x_boot, y_boot, x_oob, y_oob = draw_boot(x_train, y_train)
        tree = build_tree(
            x_boot,
            y_boot,
            max_features,
            max_depth,
            min_samples_split
        )
        tree_ls.append(tree)
        oob_error = oob_score(tree, x_oob, y_oob)
        oob_ls.append(oob_error)
    print("out of bag estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls


def predict_tree(tree, x_test):
    feature_idx = tree["feature_idx"]

    if x_test[feature_idx] <= tree["split_point"]:
        if type(tree["left_split"]) == dict:
            return predict_tree(
                tree["left_split"],
                x_test
            )
        else:
            value = tree["left_split"]
            return value
    else:
        if type(tree["right_split"]) == dict:
            return predict_tree(
                tree["right_split"],
                x_test
            )
        else:
            return tree["right_split"]
        
