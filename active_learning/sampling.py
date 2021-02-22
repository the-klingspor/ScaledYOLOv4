import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.utils.extmath import stable_cumsum


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


def kmpp(features, scores, n_clusters, paths):
    queries, query_paths = sample(features, scores, n_clusters)
    return


def core_set(features, scores, n_clusters, paths):
    return


def sample(X, scores, n_clusters, n_local_trials=None, randomized=False):
    n_samples, n_features = X.shape
    x_squared_norms = np.sum(features**2, axis=1)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    center_ids = np.empty(n_clusters, dtype=np.int)

    assert x_squared_norms is not None, 'x_squared_norms None in sample'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center according to given score
    if scores is not None:
        center_id = np.argmax(scores)
    # Pick first center randomly
    else:
        center_id = np.random.randint(n_samples)
        # if all elements have the same score, scoring is basically ignored
        scores = np.ones(n_samples)
    centers[0] = X[center_id]
    center_ids[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms)
    closest_dist = np.squeeze(closest_dist)
    # Potential is minimum distance to current cluster centers multiplied with
    # the score
    current_pot = closest_dist * scores

    rng = np.random.default_rng()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        if randomized:  # kmpp
            # Choose center candidates by sampling with probability proportional
            # to the distance to the current potential
            p = current_pot / np.sum(current_pot)
            candidate_ids = rng.choice(n_samples, n_local_trials, replace=False,
                                       p=p)

            # Choose candidate with highest current potential
            candidates_pot = current_pot[candidate_ids]
            best_candidate = candidate_ids[np.argmax(candidates_pot)]
        else:  # core-set
            best_candidate = np.argmax(current_pot)

        # Compute distances to center candidates
        distance_to_best_candidate = euclidean_distances(
            X[best_candidate, np.newaxis], X, Y_norm_squared=x_squared_norms)
        distance_to_best_candidate = np.squeeze(distance_to_best_candidate)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist, distance_to_best_candidate, out=closest_dist)
        current_pot = closest_dist * scores

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        center_ids[c] = best_candidate

    return centers, center_ids


if __name__ == '__main__':
    features = np.array([[2.0, 4.0], [1.0, 5.0], [4.0, 3.0], [1.0, 2.0],
                         [4.0, 2.0], [5.0, 9.0]])
    scores = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.5])

    centers, center_ids = sample(features, scores, 4)
    print(centers)

    centers, center_ids = sample(features, scores, 4, randomized=True)
    print(centers)
