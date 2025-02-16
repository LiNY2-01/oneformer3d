from oneformer3d.spt.transforms import Transform
from oneformer3d.spt.utils.neighbors import knn_1, inliers_split, \
    outliers_split


__all__ = ['KNN'] #, 'Inliers', 'Outliers']


class KNN(Transform):
    """K-NN search for each point in Data.
 
    Neighbors and corresponding distances are stored in
    `Data.neighbor_index` and `Data.neighbor_distance`, respectively.

    To accelerate search, neighbors are searched within a maximum radius
    of each point. This may result in points having less-than-expected
    neighbors (missing neighbors are indicated by -1 indices). The
    `oversample` mechanism allows for oversampling the found neighbors
    to replace the missing ones.

    :param k: int
        Number of neighbors to search for
    :param r_max: float
        Radius within which neighbors are searched around each point
    :param oversample: bool
        Whether partial neighborhoods should be oversampled to reach
        the target `k` neighbors per point
    :param self_is_neighbor: bool
        Whether each point should be considered as its own nearest
        neighbor or should be excluded from the search
    :param verbose: bool
    """

    _NO_REPR = ['verbose']

    def __init__(
            self, k=50, r_max=1, oversample=False, self_is_neighbor=False,
            verbose=False):
        self.k = k
        self.r_max = r_max
        self.oversample = oversample
        self.self_is_neighbor = self_is_neighbor
        self.verbose = verbose

    def _process(self, data):
        neighbors, distances = knn_1(
            data.pos,
            self.k,
            r_max=self.r_max,
            batch=data.batch,
            oversample=self.oversample,
            self_is_neighbor=self.self_is_neighbor,
            verbose=self.verbose)
        data.neighbor_index = neighbors
        data.neighbor_distance = distances
        return data
