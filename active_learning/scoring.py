import numpy as np


def entropy_scores(pred, aggr="max"):
    """
    Compute the entropy of the given object detection prediction. "aggr" is
    the strategy used for aggregating the entropy of bounding box predictions.
    The available strategies are "max", "sum" and "average".

    :param pred: Array of shape [n_bboxs, xywh + objectness + n_classes]
        The predictions of the network used to determine their entropy.
    :param aggr: String, default="max". One of {"max", "avg"}
        The aggregate function to be used.
    :return: Float
        The entropy of the prediction for the given aggregate function.
    """
    # extract the class probabilities and the objectness
    cl_probs = pred[:, 4:]
    entropy_arr = entropy(cl_probs, axis=1)

    # aggregate the entropy values based on the given mode
    entropy_score = apply_aggr(entropy_arr, aggr=aggr)

    return entropy_score


def mutual_info_scores(pred_student, pred_teacher, aggr="max"):
    """
    Compute the mutual information of the student and teacher predictions using
    the specified aggregate function. The mutual information is defined as
    the entropy of the average of predictions minus the average of individual
    entropy values of student and teacher.
    "aggr" is the strategy used for aggregating the entropy of bounding box
    predictions. The available strategies are "max" and "average".

    :param pred_student: Array of shape [n_bboxs, xywh + objectness + n_classes]
        The predictions of the student network.
    :param pred_teacher: Array of shape [n_bboxs, xywh + objectness + n_classes]
        The predictions of the teacher network.
    :param aggr: String, default="max". One of {"max", "avg"}
        The aggregate function to be used.
    :return: Float
        The mutual information between the student and teacher predictions.
    """
    cl_probs_student = pred_student[:, 4:]
    cl_probs_teacher = pred_teacher[:, 4:]

    cl_probs_avg = (cl_probs_student + cl_probs_teacher) / 2.0
    entropy_avg = entropy(cl_probs_avg, axis=1)

    entropy_student = entropy(cl_probs_student, axis=1)
    entropy_teacher = entropy(cl_probs_teacher, axis=1)
    avg_of_entropy = (entropy_student + entropy_teacher) / 2.0

    mutual_info_arr = entropy_avg - avg_of_entropy

    # aggregate the entropy values based on the given mode
    mutual_info = apply_aggr(mutual_info_arr, aggr=aggr)

    return mutual_info


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


def entropy(arr, axis=0, eps=1e-7):
    """
    Compute the entropy along the given axis.

    :param arr: np array of floats
    :param axis: int, default = 0
    :param eps: float, default = 1e-7
        Epsilon to prevent logarithm for zero values.
    :return: The array with
    """
    return -np.sum(arr * np.log2(arr+eps), axis=axis)



