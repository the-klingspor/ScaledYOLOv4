import numpy as np


def entropy_scores(pred, aggr="sum", bb=True):
    """
    Compute the entropy of the given object detection prediction. "aggr" is
    the strategy used for aggregating the entropy of bounding box predictions.
    The available strategies are "max", "sum" and "average".

    :param pred: Array of shape [n_bboxs, xywh + objectness + n_classes]
        The predictions of the network used to determine their entropy.
    :param aggr: String, default="sum". One of {"max", "avg", "sum"}
        The aggregate function to be used.
    :return: Float
        The entropy of the prediction for the given aggregate function.
    """
    # extract the class probabilities and the objectness
    cl_probs = pred[:, 4:]
    if not bb:  # for easy tests without bounding box predictions
        cl_probs = pred
    entropy_arr = entropy(cl_probs)

    # aggregate the entropy values based on the given mode
    entropy_score = apply_aggr(entropy_arr, aggr=aggr)

    return entropy_score


def mutual_info_scores(pred_student, pred_teacher, aggr="avg", bb=True):
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
    :param aggr: String, default="avg". One of {"max", "avg", "sum}
        The aggregate function to be used.
    :return: Float
        The mutual information between the student and teacher predictions.
    """
    cl_probs_student = pred_student[:, 4:]
    cl_probs_teacher = pred_teacher[:, 4:]
    if not bb:  # for easy tests without bounding box preds
        cl_probs_student = pred_student
        cl_probs_teacher = pred_teacher
    cl_probs_avg = (cl_probs_student + cl_probs_teacher) / 2.0
    entropy_avg = entropy(cl_probs_avg)

    entropy_student = entropy(cl_probs_student)
    entropy_teacher = entropy(cl_probs_teacher)
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


def entropy(arr, eps=1e-10):
    """
    Compute the element-wise entropy by interpreting every entry as Bernoulli
    random variable.

    :param arr: np array of floats
    :param eps: float, default = 1e-10
        Epsilon to prevent logarithm for zero values.
    :return: The array with the entropy in every element
    """
    return -(arr * np.log2(arr + eps) + (1 - arr) * np.log2(1 - arr + eps))


if __name__ == '__main__':
    pred_student = np.array([[0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
    pred_teacher = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    pred_student2 = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    print(entropy(pred_student))
    print(entropy(pred_teacher))
    print(entropy(pred_student2))
    print()
    print(entropy_scores(pred_student, bb=False))
    print(entropy_scores(pred_teacher, bb=False))
    print()
    print(mutual_info_scores(pred_student, pred_teacher, bb=False))
    print(mutual_info_scores(pred_student2, pred_teacher, bb=False))
    print(mutual_info_scores(pred_student, pred_student2, bb=False))
    print(mutual_info_scores(pred_student, pred_student, bb=False))

    pred_student = np.array([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
    pred_teacher = np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])

    print(mutual_info_scores(pred_student, pred_teacher, bb=False))


