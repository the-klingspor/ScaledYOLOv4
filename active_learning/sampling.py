

def top(scores, paths):
    """
    Order the paths according to their scores in ascending order. Return the
    sorted list of paths.

    :param scores: list of float
        A list containing the active learning scores of a set of images.
    :param paths: list of strings
        A list containing the paths to the respective images.
    :return: list of strings
        The list of paths, but sorted in ascending order according to the given
        scores.
    """
    paths_sorted = [p for s, p in sorted(zip(scores, paths), reverse=True)]
    return paths_sorted


def kmpp(scores, paths):
    return


def core_set(scores, paths):
    return
