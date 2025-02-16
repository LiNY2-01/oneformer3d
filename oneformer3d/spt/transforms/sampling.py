import re
import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from oneformer3d.spt.utils import fast_randperm, sparse_sample, scatter_pca, sanitize_keys
from oneformer3d.spt.transforms import Transform
from oneformer3d.spt.data import Data, NAG, NAGBatch, CSRData, InstanceData, Cluster
from oneformer3d.spt.utils.histogram import atomic_to_histogram


__all__ = [
    'Shuffle', 'SaveNodeIndex', 'NAGSaveNodeIndex', 'GridSampling3D',]
    # 'SampleXYTiling', 'SampleRecursiveMainXYAxisTiling', 'SampleSubNodes',
    # 'SampleKHopSubgraphs', 'SampleRadiusSubgraphs', 'SampleSegments',
    # 'SampleEdges', 'RestrictSize', 'NAGRestrictSize']


class Shuffle(Transform):
    """Shuffle the order of points in a Data object."""

    def _process(self, data):
        idx = fast_randperm(data.num_points, device=data.device)
        return data.select(idx, update_sub=False, update_super=False)


class SaveNodeIndex(Transform):
    """Adds the index of the nodes to the Data object attributes. This
    allows tracking nodes from the output back to the input Data object.
    """

    DEFAULT_KEY = 'node_id'

    def __init__(self, key=None):
        self.key = key if key is not None else self.DEFAULT_KEY

    def _process(self, data):
        idx = torch.arange(0, data.pos.shape[0], device=data.device)
        setattr(data, self.key, idx)
        return data


class NAGSaveNodeIndex(SaveNodeIndex):
    """SaveNodeIndex, applied to each NAG level.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        transform = SaveNodeIndex(key=self.key)
        for i_level in range(nag.num_levels):
            nag._list[i_level] = transform(nag._list[i_level])
        return nag

# TODO: migrate

class GridSampling3D(Transform):
    """ Clusters 3D points into voxels with size :attr:`size`.

    By default, some special keys undergo dedicated grouping mechanisms.
    The `_VOTING_KEYS=['y', 'super_index', 'is_val']` keys are grouped
    by their majority label. The `_INSTANCE_KEYS=['obj', 'obj_pred']`
    keys are grouped into an `InstanceData`, which stores all
    instance/panoptic overlap data values in CSR format. The
    `_CLUSTER_KEYS=['point_id']` keys are grouped into a `Cluster`
    object, which stores indices of child elements for parent clusters
    in CSR format. The `_LAST_KEYS=['batch', SaveNodeIndex.DEFAULT_KEY]`
    keys are by default grouped following `mode='last'`.

    Besides, for keys where a more subtle histogram mechanism is needed,
    (e.g. for 'y'), the 'hist_key' and 'hist_size' arguments can be
    used.

    Modified from: https://github.com/torch-points3d/torch-points3d

    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
        If True, it will convert the points into their associated sparse
        coordinates within the grid and store the value into a new
        `coords` attribute.
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a
        cell will be averaged. If mode is `last`, one random points per
        cell will be selected with its associated features.
    hist_key: str or List(str)
        Data attributes for which we would like to aggregate values into
        an histogram. This is typically needed when we want to aggregate
        points labels without losing the distribution, as opposed to
        majority voting.
    hist_size: str or List(str)
        Must be of same size as `hist_key`, indicates the number of
        bins for each key-histogram. This is typically needed when we
        want to aggregate points labels without losing the distribution,
        as opposed to majority voting.
    inplace: bool
        Whether the input Data object should be modified in-place
    verbose: bool
        Verbosity
    """

    _NO_REPR = ['verbose', 'inplace']

    def __init__(
            self, size, quantize_coords=False, mode="mean", hist_key=None,
            hist_size=None, inplace=False, verbose=False):

        hist_key = [] if hist_key is None else hist_key
        hist_size = [] if hist_size is None else hist_size
        hist_key = [hist_key] if isinstance(hist_key, str) else hist_key
        hist_size = [hist_size] if isinstance(hist_size, int) else hist_size

        assert isinstance(hist_key, list)
        assert isinstance(hist_size, list)
        assert len(hist_key) == len(hist_size)

        self.grid_size = size
        self.quantize_coords = quantize_coords
        self.mode = mode
        self.bins = {k: v for k, v in zip(hist_key, hist_size)}
        self.inplace = inplace

        if verbose:
            print(
                f"If you need to keep track of the position of your points, "
                f"use SaveNodeIndex transform before using "
                f"{self.__class__.__name__}.")

            if self.mode == "last":
                print(
                    "The tensors within data will be shuffled each time this "
                    "transform is applied. Be careful that if an attribute "
                    "doesn't have the size of num_nodes, it won't be shuffled")

    def _process(self, data_in):
        # In-place option will modify the input Data object directly
        data = data_in if self.inplace else data_in.clone()

        # If the aggregation mode is 'last', shuffle the points order.
        # Note that voxelization of point attributes will be stochastic
        if self.mode == 'last':
            data = Shuffle()(data)

        # Convert point coordinates to the voxel grid coordinates
        coords = torch.round((data.pos) / self.grid_size)

        # Match each point with a voxel identifier
        if 'batch' not in data:
            cluster = grid_cluster(coords, torch.ones(3, device=coords.device))
        else:
            cluster = voxel_grid(coords, data.batch, 1)

        # Reindex the clusters to make sure the indices used are
        # consecutive. Basically, we do not want cluster indices to span
        # [0, i_max] without all in-between indices to be used, because
        # this will affect the speed and output size of torch_scatter
        # operations
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        # Perform voxel aggregation
        data = _group_data(
            data, cluster, unique_pos_indices, mode=self.mode, bins=self.bins)

        # Optionally convert quantize the coordinates. This is useful
        # for sparse convolution models
        if self.quantize_coords:
            data.coords = coords[unique_pos_indices].int()

        # Save the grid size in the Data attributes
        data.grid_size = torch.tensor([self.grid_size])

        return data


def _group_data(
        data, cluster=None, unique_pos_indices=None, mode="mean",
        skip_keys=None, bins={}):
    """Group data based on indices in cluster. The option ``mode``
    controls how data gets aggregated within each cluster.

    By default, some special keys undergo dedicated grouping mechanisms.
    The `_VOTING_KEYS=['y', 'super_index', 'is_val']` keys are grouped
    by their majority label. The `_INSTANCE_KEYS=['obj', 'obj_pred']`
    keys are grouped into an `InstanceData`, which stores all
    instance/panoptic overlap data values in CSR format. The
    `_CLUSTER_KEYS=['point_id']` keys are grouped into a `Cluster`
    object, which stores indices of child elements for parent clusters
    in CSR format. The `_LAST_KEYS=['batch', SaveNodeIndex.DEFAULT_KEY]`
    keys are by default grouped following `mode='last'`.

    Besides, for keys where a more subtle histogram mechanism is needed,
    (e.g. for 'y'), the 'bins' argument can be used.

    Warning: this function modifies the input Data object in-place.

    :param data : Data
    :param cluster : Tensor
        Tensor of the same size as the number of points in data. Each
        element is the cluster index of that point.
    :param unique_pos_indices : Tensor
        Tensor containing one index per cluster, this index will be used
        to select features and labels.
    :param mode : str
        Option to select how the features and labels for each voxel is
        computed. Can be ``last`` or ``mean``. ``last`` selects the last
        point falling in a voxel as the representative, ``mean`` takes
        the average.
    :param skip_keys: list
        Keys of attributes to skip in the grouping.
    :param bins: dict
        Dictionary holding ``{'key': n_bins}`` where ``key`` is a Data
        attribute for which we would like to aggregate values into an
        histogram and ``n_bins`` accounts for the corresponding number
        of bins. This is typically needed when we want to aggregate
        point labels without losing the distribution, as opposed to
        majority voting.
    """
    skip_keys = sanitize_keys(skip_keys, default=[])

    # Keys for which voxel aggregation will be based on majority voting
    _VOTING_KEYS = ['y', 'super_index', 'is_val']

    # Keys for which voxel aggregation will use an InstanceData object,
    # which store all input information in CSR format
    _INSTANCE_KEYS = ['obj', 'obj_pred']

    # Keys for which voxel aggregation will use a Cluster object, which 
    # store all input information in CSR format
    _CLUSTER_KEYS = ['sub']

    # Keys for which voxel aggregation will be based on majority voting
    _LAST_KEYS = ['batch', SaveNodeIndex.DEFAULT_KEY]

    # Keys to be treated as normal vectors, for which the unit-norm must
    # be preserved
    _NORMAL_KEYS = ['normal']

    # Supported mode for aggregation
    _MODES = ['mean', 'last']
    assert mode in _MODES
    if mode == "mean" and cluster is None:
        raise ValueError(
            "In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError(
            "In last mode the unique_pos_indices argument needs to be specified")

    # Save the number of nodes here because the subsequent in-place
    # modifications will affect it
    num_nodes = data.num_nodes

    # Aggregate Data attributes for same-cluster points
    for key, item in data:

        # `skip_keys` are not aggregated
        if key in skip_keys:
            continue

        # Edges cannot be aggregated
        if bool(re.search('edge', key)):
            raise NotImplementedError("Edges not supported. Wrong data type.")

        # For instance labels grouped into an InstanceData. Supports
        # input instance labels either as InstanceData or as a simple
        # index tensor
        if key in _INSTANCE_KEYS:
            if isinstance(item, InstanceData):
                data[key] = item.merge(cluster)
            else:
                count = torch.ones_like(item)
                y = data.y if getattr(data, 'y', None) is not None \
                    else torch.zeros_like(item)
                data[key] = InstanceData(cluster, item, count, y, dense=True)
            continue
        
        # For point indices to be grouped in Cluster. This allows 
        # backtracking full-resolution point indices to the voxels
        if key in _CLUSTER_KEYS:
            if (isinstance(item, torch.Tensor) and item.dim() == 1
                    and not item.is_floating_point()):
                data[key] = Cluster(cluster, item, dense=True)
            else:
                raise NotImplementedError(
                    f"Cannot merge '{key}' with data type: {type(item)} into "
                    f"a Cluster object. Only supports 1D Tensor of integers.")
            continue

        # TODO: adapt to make use of CSRData batching ?
        if isinstance(item, CSRData):
            raise NotImplementedError(
                f"Cannot merge '{key}' with data type: {type(item)}")

        # Only torch.Tensor attributes of size Data.num_nodes are
        # considered for aggregation
        if not torch.is_tensor(item) or item.size(0) != num_nodes:
            continue

        # For 'last' mode, use unique_pos_indices to pick values
        # from a single point within each cluster. The same behavior
        # is expected for the _LAST_KEYS
        if mode == 'last' or key in _LAST_KEYS:
            data[key] = item[unique_pos_indices]
            continue

        # For 'mean' mode, the attributes will be aggregated
        # depending on their nature.

        # If the attribute is a boolean, temporarily convert to integer
        # to facilitate aggregation
        is_item_bool = item.dtype == torch.bool
        if is_item_bool:
            item = item.int()

        # For keys requiring a voting scheme or a histogram
        if key in _VOTING_KEYS or key in bins.keys():
            voting = key not in bins.keys()
            n_bins = item.max() + 1 if voting else bins[key]
            hist = atomic_to_histogram(item, cluster, n_bins=n_bins)
            data[key] = hist.argmax(dim=-1) if voting else hist

        # Standard behavior, where attributes are simply
        # averaged across the clusters
        else:
            data[key] = scatter_mean(item, cluster, dim=0)

        # For normals, make sure to re-normalize the mean-normal
        if key in _NORMAL_KEYS:
            data[key] = data[key] / data[key].norm(dim=1).view(-1, 1)

        # Convert back to boolean if need be
        if is_item_bool:
            data[key] = data[key].bool()

    return data

    """Randomly sample nodes and edges to restrict their number within
    given limits. This is useful for stabilizing memory use of the
    model.

    :param num_nodes: int
        Maximum number of nodes. If the input has more, a subset of
        `num_nodes` nodes will be randomly sampled. No sampling if <=0
    :param num_edges: int
        Maximum number of edges. If the input has more, a subset of
        `num_edges` edges will be randomly sampled. No sampling if <=0
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='1+', num_nodes=0, num_edges=0):
        assert isinstance(level, (int, str))
        assert isinstance(num_nodes, (int, list))
        assert isinstance(num_edges, (int, list))
        self.level = level
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def _process(self, nag):

        # If 'level' is an int, we only need to process a single level
        if isinstance(self.level, int):
            return self._restrict_level(
                nag, self.level, self.num_nodes, self.num_edges)

        # If 'level' covers multiple levels, iteratively process levels
        level_num_nodes = [-1] * nag.num_levels
        level_num_edges = [-1] * nag.num_levels

        if self.level == 'all':
            level_num_nodes = self.num_nodes \
                if isinstance(self.num_nodes, list) \
                else [self.num_nodes] * nag.num_levels
            level_num_edges = self.num_edges \
                if isinstance(self.num_edges, list) \
                else [self.num_edges] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_num_nodes[i:] = self.num_nodes \
                if isinstance(self.num_nodes, list) \
                else [self.num_nodes] * (nag.num_levels - i)
            level_num_edges[i:] = self.num_edges \
                if isinstance(self.num_edges, list) \
                else [self.num_edges] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_num_nodes[:i] = self.num_nodes \
                if isinstance(self.num_nodes, list) \
                else [self.num_nodes] * i
            level_num_edges[:i] = self.num_edges \
                if isinstance(self.num_edges, list) \
                else [self.num_edges] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        for i_level, (num_nodes, num_edges) in enumerate(zip(
                level_num_nodes, level_num_edges)):
            nag = self._restrict_level(nag, i_level, num_nodes, num_edges)

        return nag

    @staticmethod
    def _restrict_level(nag, i_level, num_nodes, num_edges):

        if nag[i_level].num_nodes > num_nodes and num_nodes > 0:
            weights = torch.ones(nag[i_level].num_nodes, device=nag.device)
            idx = torch.multinomial(weights, num_nodes, replacement=False)
            nag = nag.select(i_level, idx)

        if nag[i_level].num_edges > num_edges and num_edges > 0:
            weights = torch.ones(nag[i_level].num_edges, device=nag.device)
            idx = torch.multinomial(weights, num_edges, replacement=False)

            nag[i_level].edge_index = nag[i_level].edge_index[:, idx]
            if nag[i_level].has_edge_attr:
                nag[i_level].edge_attr = nag[i_level].edge_attr[idx]
            for key in nag[i_level].edge_keys:
                nag[i_level][key] = nag[i_level][key][idx]

        return nag