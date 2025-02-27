import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

from oneformer3d.spt.transforms import PointFeatures, GroundElevation, AdjacencyGraph, ConnectIsolated, AddKeysTo, \
    CutPursuitPartition
from oneformer3d.spt.transforms.device import DataTo
from oneformer3d.spt.transforms.neighbors import KNN
from oneformer3d.spt.transforms.sampling import GridSampling3D
from oneformer3d.spt.data import Data
from oneformer3d.spt.transforms import SaveNodeIndex

@TRANSFORMS.register_module()
class ExtractSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True,
                 voxel=0.02,
                 knn=45,knn_r=2,
                 knn_step=-1,knn_min_search=25,
                 ground_threshold=1.5,ground_scale=4.0,
                 pcp_regularization= [0.01, 0.1, 0.5],
                 pcp_spatial_weight=[1e-1, 1e-1, 1e-1],
                 pcp_cutoff=[10, 10, 10],
                 pcp_k_adjacency=10,pcp_w_adjacency=1,
                 pcp_iterations=15
                 ):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
        self.voxel = voxel
        self.knn = knn
        self.knn_r = knn_r
        self.knn_step = knn_step
        self.knn_min_search = knn_min_search
        self.ground_threshold = ground_threshold
        self.ground_scale =ground_scale
        self.pcp_regularization=pcp_regularization
        self.pcp_spatial_weight=pcp_spatial_weight
        self.pcp_cutoff=pcp_cutoff
        self.pcp_k_adjacency=pcp_k_adjacency
        self.pcp_w_adjacency=pcp_w_adjacency
        self.pcp_iterations=pcp_iterations


 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        # pts_coord = input_dict['points'].coord.numpy()
        # pts_color = input_dict['points'].color.numpy()

        pts_coord = input_dict['points'].coord
        pts_color = input_dict['points'].color
        pos_offset = torch.zeros_like(pts_coord[0])
        spt_data = Data(pos=pts_coord,pos_offset=pos_offset,rgb=pts_color)

        spt_data = SaveNodeIndex(key='sub')(spt_data)
        spt_data = DataTo('cuda')(spt_data)
        spt_data = GridSampling3D(size=self.voxel)(spt_data)
        spt_data = KNN(k=self.knn,r_max=self.knn_r)(spt_data)
        spt_data = DataTo('cpu')(spt_data)
        spt_data = PointFeatures(
            keys = [
                'rgb','linearity','planarity',
                'scattering','verticality','elevation',
            ],
            k_min=1,k_step=self.knn_step,k_min_search=self.knn_min_search)(spt_data)
        spt_data = GroundElevation(scale=self.ground_scale,z_threshold=self.ground_threshold)(spt_data)
        spt_data = DataTo('cuda')(spt_data)
        spt_data = AdjacencyGraph(k=self.pcp_k_adjacency,w=self.pcp_w_adjacency)(spt_data)
        spt_data = ConnectIsolated(k=1)(spt_data)
        spt_data = DataTo('cpu')(spt_data)
        spt_data = AddKeysTo(
            keys = [
                'rgb','linearity','planarity',
                'scattering','verticality','elevation',
            ],to='x',delete_after=False,)(spt_data)
        spt_data = CutPursuitPartition(
            regularization=self.pcp_regularization,
            spatial_weight=self.pcp_spatial_weight,
            k_adjacency=self.pcp_k_adjacency,
            cutoff=self.pcp_cutoff,
            iterations=self.pcp_iterations,
            parallel=True,
            verbose=False,
        )(spt_data)
        # spt_data :NAG
        sp_index = spt_data.get_super_index(2,-1)




        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(pts_coord)
        # point_cloud.colors = o3d.utility.Vector3dVector(pts_color)
        #
        # # get super point mask
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.03)
        #
        # vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        # faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        # superpoints = segmentator.segment_mesh(vertices, faces).numpy()
        # superpoints = [0]
        input_dict['sp_pts_mask'] = sp_index

        return input_dict