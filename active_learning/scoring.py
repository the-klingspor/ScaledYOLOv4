import numpy as np


def entropy_score(pred, aggr="sum"):
    """
    Compute the entropy of the given object detection prediction. "aggr" is
    the strategy used for aggregating the entropy of bounding box predictions.
    The available strategies are "max", "sum" and "average".

    :param pred: Array of shape [n_bboxs] or [n_bboxes, n_classes]
        The predictions of the network used to determine their entropy.
    :param aggr: String, default="sum". One of {"max", "avg", "sum"}
        The aggregate function to be used.
    :return: Float
        The entropy of the prediction for the given aggregate function.
    """
    x = pred
    if x.ndim == 1:  # single class or objectness
        entropy_arr = entropy_bern(x)
    else:  # multiple classes
        entropy_arr = entropy(x)

    # aggregate the entropy values based on the given mode
    entropy_score = apply_aggr(entropy_arr, aggr=aggr)

    return entropy_score


def v_entropy_scores(x):
    """
    Compute the entropy score for every element in x and
    :param x:
    :return:
    """
    n = len(x)
    entropy_scores = []
    for i in range(n):
        entropy_scores.append(entropy_score(x[i]))
    return entropy_scores


def jensen_shannon_score(pred_student, pred_teacher, aggr="avg"):
    """
    Compute the Jensen-Shannon divergence of the student and teacher predictions
    using the specified aggregate function. The Jensen-Shannon divergence is
    defined as the entropy of the average of predictions minus the average of
    individual entropy values of student and teacher.
    "aggr" is the strategy used for aggregating the entropy of bounding box
    predictions. The available strategies are "max", "avg" and "sum".

    :param pred_student: Array of shape [n_bboxs] or [n_bboxs, n_classes]
        The predictions of the student network.
    :param pred_teacher: Array of the same shape as pred_student
        The predictions of the teacher network.
    :param aggr: String, default="avg". One of {"max", "avg", "sum}
        The aggregate function to be used.
    :return: Float
        The Jensen-Shannon divergence between the student and teacher
        predictions.
    """
    pred_avg = (pred_student + pred_teacher) / 2.0
    if pred_student.ndim == 1:  # single class or objectness
        entropy_avg = entropy_bern(pred_avg)
        entropy_student = entropy_bern(pred_student)
        entropy_teacher = entropy_bern(pred_teacher)
    else:  # multiple classes
        entropy_avg = entropy(pred_avg)
        entropy_student = entropy(pred_student)
        entropy_teacher = entropy(pred_teacher)

    avg_of_entropy = (entropy_student + entropy_teacher) / 2.0

    jensen_shannon_arr = entropy_avg - avg_of_entropy

    # aggregate the entropy values based on the given mode
    jensen_shannon = apply_aggr(jensen_shannon_arr, aggr=aggr)

    return jensen_shannon


v_jensen_shannon_scores = np.vectorize(jensen_shannon_score)


def apply_aggr(scores, aggr):
    if aggr == "max":
        score = np.max(scores)
    elif aggr == "sum":
        score = np.sum(scores)
    elif aggr == "avg":
        score = np.mean(scores)
    else:
        raise AssertionError("Aggregation scheme is invalid.")
    return score


def entropy_bern(arr, eps=1e-6):
    """
    Compute the element-wise entropy by interpreting every entry as Bernoulli
    random variable.

    :param arr: np array of floats
    :param eps: float, default = 1e-6
        Epsilon to prevent logarithm for zero values.
    :return: The array with the entropy in every element
    """
    x = np.array(arr, dtype=np.float)
    x = np.clip(x, eps, 1.0 - eps)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def entropy(arr, eps=1e-6, axis=1):
    """
    Compute the entropy along the given axis.

    :param arr: np array of floats [n_values, n_classes]
    :param eps: float, default = 1e-6
        Epsilon to prevent logarithm for zero values.
    :return: The array with the entropy in every element
    """
    x = np.array(arr, dtype=np.float)
    x = np.clip(x, eps, 1.0 - eps)
    entropy_vals = -(x * np.log2(x))
    return np.sum(entropy_vals, axis=axis)


def scale_lin(x, conf_thres=0.1, target=0.5):
    """
    Scale x so that conf_thres -> target with a linear function. Sets conf_thres ->
    0.5 and interpolates linearly for values above and below.
    :param x: float
    :param conf_thres: float, default=0.1
        The value to map to the target.
    :param target: float, default=0.5
        The value to which conf_thres should be mapped
    :return: float
        Transformed x.
    """
    if x <= conf_thres:
        return (target / conf_thres) * x
    else:
        return target + ((1 - target) / (1 - conf_thres)) * (x - conf_thres)


v_scale_lin = np.vectorize(scale_lin)


def scale_sigmoid(x, conf_thres=0.1, k=0.1):
    """
    Scale x so that conf_thres -> 0.5 with a sigmoid function. This gives poor
    approximations for conf_tres close to 0 and 1.
    :param x: float
    :param conf_thres: float, default=0.1
        The value to map to 0.5.
    :param k: float, default=0.1
        The slope parameter of the sigmoid function.
    :return: Float
        Transformed x.
    """
    s = 1 / (1 + np.exp(-(x - conf_thres) / k))
    return s


v_scale_sigmoid = np.vectorize(scale_sigmoid)


def solve_minkowski(conf_thres, p=1.0, min=1e-10, max=1e5, eps=1e-5):
    """
    Find the value p for which (conf_thres - 1, 0.5) lies on the unit-circle of
    the Minkowski-p distance.

    :param conf_thres:
    :param p: float, default = 1.0
        Starting value for the Minkowski parameter.
    :param min: float, default = 1e-10
        Left border of possible p params.
    :param max: float, default = 1e5
        Right border of possible p params.
    :param eps: float, default = 1e-5
        How close the computed Minkowski distance for the current p has to be
        to the unit circle to be accepted as final result.
    :return:
    """
    mink_dist = abs(conf_thres - 1)**p + 0.5**p
    if abs(mink_dist - 1) < eps:
        return p
    elif mink_dist < 1:
        p_new = (p + min) / 2
        return solve_minkowski(conf_thres, p=p_new, min=min, max=p, eps=eps)
    else:
        p_new = (p + max) / 2
        return solve_minkowski(conf_thres, p=p_new, min=p, max=max, eps=eps)


def v_scale_mink(x, conf_thres=0.1):
    """
    Scales the array x to [0,1] such that the scaled result of every value lies
    on a unit circle of the Minkowski-p distance such that (conf_thres - 1, 0.5)
    lies on it as well.

    :param x: Array of floats
    :param conf_thres: float, default=0.1
        The value to scale to 0.5
    :return: Array of floats
        The scaled x values.
    """
    p = solve_minkowski(conf_thres)
    x_mink = (1 - np.abs(x - 1)**p)**(1 / p)
    return x_mink


if __name__ == '__main__':
    pred_student = np.array([[0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
    pred_teacher = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    pred_student2 = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    print("Entropy for every BB:")
    print(entropy(pred_student))
    print(entropy(pred_teacher))
    print(entropy(pred_student2))
    print("\nElementwise Bernoulli entropy:")
    print(entropy_bern(pred_student[:, 0]))
    print(entropy_bern(pred_teacher[:, 0]))
    print("\nEntropy scores with sum aggregation:")
    print(entropy_score(pred_student))
    print(entropy_score(pred_teacher))
    print("\nJensen-Shannon divergence with avg aggregation:")
    print(jensen_shannon_score(pred_student, pred_teacher))
    print(jensen_shannon_score(pred_student2, pred_teacher))
    print(jensen_shannon_score(pred_student, pred_student2))
    print(jensen_shannon_score(pred_student, pred_student))

    pred_student = np.array([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
    pred_teacher = np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])

    print(jensen_shannon_score(pred_student, pred_teacher))

    print("\nTests with Minkowski scaling:")
    obj = np.array([0.1, 0.001, 0.5])
    conf_thres = 0.1
    obj_mink = v_scale_mink(obj, conf_thres)
    print(obj_mink)
    print(entropy_bern(obj))
    print(entropy_bern(obj_mink))


