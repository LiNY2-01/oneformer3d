import numpy as np
import pgeof
import torch
from torch_scatter import scatter_mean, scatter_std

import oneformer3d.spt as src
from oneformer3d.spt.data import NAG
from oneformer3d.spt.utils import print_tensor_info, scatter_mean_orientation, POINT_FEATURES, \
    SEGMENT_BASE_FEATURES, sanitize_keys
from .transforms import Transform

__all__ = [
    'AdjacencyGraph', 'SegmentFeatures',
    # 'DelaunayHorizontalGraph',
    # 'RadiusHorizontalGraph', 'OnTheFlyHorizontalEdgeFeatures',
    # 'OnTheFlyVerticalEdgeFeatures', 'NAGAddSelfLoops',
    'ConnectIsolated',
    'NodeSize']


class AdjacencyGraph(Transform):
    """Create the adjacency graph in `edge_index` and `edge_attr` based
    on the `Data.neighbor_index` and `Data.neighbor_distance`.

    NB: the produced graph is directed wrt Pytorch Geometric, but
    `CutPursuitPartition` happily takes it as an input.

    :param k: int
        Number of neighbors to consider for the adjacency graph. In view
        of calling `CutPursuitPartition`, note the higher the number of
        neighbors/edges per node, the longer the partition computation.
        Yet, if the number of neighbors is not sufficient, the
    :param w: float
        Scalar used to modulate the edge weight. If `w <= 0`, all edges
        will have a weight of 1. Otherwise, edges weights will follow:
        ```1 / (w + neighbor_distance / neighbor_distance.mean())```
    """

    def __init__(self, k=10, w=-1):
        self.k = k
        self.w = w

    def _process(self, data):
        assert data.has_neighbors, \
            "Data must have 'neighbor_index' attribute to allow adjacency " \
            "graph construction."
        assert getattr(data, 'neighbor_distance', None) is not None \
               or self.w <= 0, \
            "Data must have 'neighbor_distance' attribute to allow adjacency " \
            "graph construction."
        assert self.k <= data.neighbor_index.shape[1]

        # Compute source and target indices based on neighbors
        source = torch.arange(
            data.num_nodes, device=data.device).repeat_interleave(self.k)
        target = data.neighbor_index[:, :self.k].flatten()

        # Account for -1 neighbors and delete corresponding edges
        mask = target >= 0
        source = source[mask]
        target = target[mask]

        # Save edges and edge features in data
        data.edge_index = torch.stack((source, target))
        if self.w > 0:
            # Recover the neighbor distances and apply the masking
            distances = data.neighbor_distance[:, :self.k].flatten()[mask]
            data.edge_attr = 1 / (self.w + distances / distances.mean())
        else:
            data.edge_attr = torch.ones_like(source, dtype=torch.float)

        return data



"""
RadiusHorizontalGraph
SegmentFeatures
may not used
"""

class SegmentFeatures(Transform):
    """Compute segment features for all the NAG levels except its first
    (i.e. the 0-level). These are handcrafted node features that will be
    saved in the node attributes. To make use of those at training time,
    remember to move them to the `x` attribute using `AddKeysTo` and
    `NAGAddKeysTo`.

    The supported feature keys are the following:
      - linearity
      - planarity
      - scattering
      - verticality
      - curvature
      - log_length
      - log_surface
      - log_volume
      - normal
      - log_size

    :param n_max: int
        Maximum number of level-0 points to sample in each cluster to
        when building node features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    :param keys: List(str), str, or None
        Features to be computed segment-wise and saved under `<key>`.
        If None, all supported features will be computed
    :param mean_keys: List(str), str, or None
        Features to be computed from the points and the segment-wise
        mean aggregation will be saved under `mean_<key>`. If None, all
        supported features will be computed
    :param std_keys: List(str), str, or None
        Features to be computed from the points and the segment-wise
        std aggregation will be saved under `std_<key>`. If None, all
        supported features will be computed
    :param strict: bool
        If True, will raise an exception if an attribute from key is
        not within the input point Data keys
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(
            self,
            n_max=32,
            n_min=5,
            keys=None,
            mean_keys=None,
            std_keys=None,
            strict=True):
        self.n_max = n_max
        self.n_min = n_min
        self.keys = sanitize_keys(keys, default=SEGMENT_BASE_FEATURES)
        self.mean_keys = sanitize_keys(mean_keys, default=POINT_FEATURES)
        self.std_keys = sanitize_keys(std_keys, default=POINT_FEATURES)
        self.strict = strict

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag = _compute_cluster_features(
                i_level,
                nag,
                n_max=self.n_max,
                n_min=self.n_min,
                keys=self.keys,
                mean_keys=self.mean_keys,
                std_keys=self.std_keys,
                strict=self.strict)
        return nag


def _compute_cluster_features(
        i_level,
        nag,
        n_max=32,
        n_min=5,
        keys=None,
        mean_keys=None,
        std_keys=None,
        strict=True):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster features on level-0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"

    keys = sanitize_keys(keys, default=SEGMENT_BASE_FEATURES)
    mean_keys = sanitize_keys(mean_keys, default=POINT_FEATURES)
    std_keys = sanitize_keys(std_keys, default=POINT_FEATURES)

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes
    device = nag.device

    # Compute how many level-0 points each level cluster contains
    sub_size = nag.get_sub_size(i_level, low=0)

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max, n_min=n_min,
        return_pointers=True)

    # Compute cluster geometric features
    xyz = nag[0].pos[idx_samples].cpu().numpy()
    nn = np.arange(idx_samples.shape[0]).astype('uint32')
    nn_ptr = ptr_samples.cpu().numpy().astype('uint32')

    # Heuristic to avoid issues when a cluster sampling is such that
    # it produces singular covariance matrix (e.g. the sampling only
    # contains the same point repeated multiple times)
    xyz = xyz + torch.rand(xyz.shape).numpy() * 1e-5

    # C++ geometric features computation on CPU
    f = pgeof.compute_features(xyz, nn, nn_ptr, 5, verbose=False)
    f = torch.from_numpy(f)

    # Recover length, surface and volume
    if 'linearity' in keys:
        data.linearity = f[:, 0].to(device).view(-1, 1)

    if 'planarity' in keys:
        data.planarity = f[:, 1].to(device).view(-1, 1)

    if 'scattering' in keys:
        data.scattering = f[:, 2].to(device).view(-1, 1)

    if 'verticality' in keys:
        data.verticality = f[:, 3].to(device).view(-1, 1)

    if 'curvature' in keys:
        data.curvature = f[:, 10].to(device).view(-1, 1)

    if 'log_length' in keys:
        data.log_length = torch.log(f[:, 7] + 1).to(device).view(-1, 1)

    if 'log_surface' in keys:
        data.log_surface = torch.log(f[:, 8] + 1).to(device).view(-1, 1)

    if 'log_volume' in keys:
        data.log_volume = torch.log(f[:, 9] + 1).to(device).view(-1, 1)

    # As a way to "stabilize" the normals' orientation, we choose to
    # express them as oriented in the z+ half-space
    if 'normal' in keys:
        data.normal = f[:, 4:7].view(-1, 3).to(device)
        data.normal[data.normal[:, 2] < 0] *= -1

    if 'log_size' in keys:
        data.log_size = (torch.log(sub_size + 1).view(-1, 1) - np.log(2)) / 10

    # Get the cluster index each poitn belongs to
    super_index = nag.get_super_index(i_level)

    # Add the mean of point attributes, identified by their key
    for key in mean_keys:
        f = getattr(nag[0], key, None)
        if f is None and strict:
            raise ValueError(f"No point key `{key}` to build 'mean_{key} key'")
        if f is None:
            continue
        if key == 'normal':
            data[f'mean_{key}'] = scatter_mean_orientation(
                nag[0][key], super_index)
        else:
            data[f'mean_{key}'] = scatter_mean(nag[0][key], super_index, dim=0)

    # Add the std of point attributes, identified by their key
    for key in std_keys:
        f = getattr(nag[0], key, None)
        if f is None and strict:
            raise ValueError(f"No point key `{key}` to build 'std_{key} key'")
        if f is None:
            continue
        data[f'std_{key}'] = scatter_std(nag[0][key], super_index, dim=0)

    # To debug sampling
    if src.is_debug_enabled():
        data.super_super_index = super_index.to(device)
        data.node_idx_samples = idx_samples.to(device)
        data.node_xyz_samples = torch.from_numpy(xyz).to(device)
        data.node_nn_samples = torch.from_numpy(nn.astype('int64')).to(device)
        data.node_nn_ptr_samples = torch.from_numpy(
            nn_ptr.astype('int64')).to(device)

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.repeat_interleave(
            torch.arange(num_nodes), end - start)
        print('\n\n' + '*' * 50)
        print(f'        cluster graph for level={i_level}')
        print('*' * 50 + '\n')
        print(f'nag: {nag}')
        print(f'data: {data}')
        print('\n* Sampling for superpoint features')
        print_tensor_info(idx_samples, name='idx_samples')
        print_tensor_info(ptr_samples, name='ptr_samples')
        print(f'all clusters have a ptr:                   '
              f'{ptr_samples.shape[0] - 1 == num_nodes}')
        print(f'all clusters received n_min+ samples:      '
              f'{(end - start).ge(n_min).all()}')
        print(f'clusters which received no sample:         '
              f'{torch.where(end == start)[0].shape[0]}/{num_nodes}')
        print(f'all points belong to the correct clusters: '
              f'{torch.equal(super_index[idx_samples], super_index_samples)}')

    # Update the i_level Data in the NAG
    nag._list[i_level] = data

    return nag




class ConnectIsolated(Transform):
    """Creates edges for isolated nodes. Each isolated node is connected
    to the `k` nearest nodes. If the Data graph contains edge features
    in `Data.edge_attr`, the new edges will receive features based on
    their length and a linear regression of the relation between
    existing edge features and their corresponding edge length.

    NB: this is an inplace operation that will modify the input data.

    :param k: int
        Number of neighbors the isolated nodes should be connected to
    """

    def __init__(self, k=1):
        self.k = k

    def _process(self, data):
        return data.connect_isolated(k=self.k)


class NodeSize(Transform):
    """Compute the number of `low`-level elements are contained in each
    segment, at each above-level. Results are save in the `node_size`
    attribute of the corresponding Data objects.

    Note: `low=-1` is accepted when level-0 has a `sub` attribute
    (i.e. level-0 points are themselves segments of `-1` level absent
    from the NAG object).

    :param low: int
        Level whose elements we want to count
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, low=0):
        assert isinstance(low, int) and low >= -1
        self.low = low

    def _process(self, nag):
        for i_level in range(self.low + 1, nag.num_levels):
            nag[i_level].node_size = nag.get_sub_size(i_level, low=self.low)
        return nag
