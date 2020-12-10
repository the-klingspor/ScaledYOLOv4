


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


def kmpp(features, scores, paths):
    return


def core_set(features, scores, paths):
    return


def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : int
        The number of seeds to choose
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.

    This implementation of kmpp has been taken from Scikit-learn's k-means
    implementation. It was adjusted to allow non-random core-set selection and
    scaling based on precomputed scores.

    Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
             Thomas Rueckstiess <ruecksti@in.tum.de>
             James Bergstra <james.bergstra@umontreal.ca>
             Jan Schlueter <scikit-learn@jan-schlueter.de>
             Nelle Varoquaux
             Peter Prettenhofer <peter.prettenhofer@gmail.com>
             Olivier Grisel <olivier.grisel@ensta.org>
             Mathieu Blondel <mathieu@mblondel.org>
             Robert Layton <robertlayton@gmail.com>
    License: BSD 3 clause
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]

    return centers