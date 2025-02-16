import copy
import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer, get_palette
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer 
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, Det3DDataSample,
                                LiDARInstance3DBoxes, PointData,
                                points_cam2img)
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.visualization import (write_obj)
import colorsys
import random
import os.path as osp

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None



@VISUALIZERS.register_module()
class Det3dInstanceVisualizer(Det3DLocalVisualizer):

    def __init__(
        self,
        name: str = 'visualizer',
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        fig_show_cfg: dict = dict(figsize=(18, 12))
    ) -> None:
        super().__init__(
            name,
            points,
            image,
            pcd_mode,
            vis_backends,
            save_dir,
            bbox_color,
            text_color,
            mask_color,
            line_width,
            frame_cfg,
            alpha,
            multi_imgs_col,
            fig_show_cfg,
        )
    def _draw_pts_ins_seg(self,
                          points: Union[Tensor, np.ndarray],
                          pts_seg: PointData,
                          palette: Optional[List[tuple]] = None,
                          ignore_index: Optional[int] = None) -> None:
        """Draw 3D semantic mask of GT or prediction.

        Args:
            points (Tensor or np.ndarray): The input point cloud to draw.
            pts_seg (:obj:`PointData`): Data structure for pixel-level
                annotations or predictions.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.
            ignore_index (int, optional): Ignore category. Defaults to None.
        """
        check_type('points', points, (np.ndarray, Tensor))

        points = tensor2ndarray(points)
        # pts_ins_seg = tensor2ndarray(pts_seg.pts_instance_mask[1])
        pts_ins_mask = tensor2ndarray(pts_seg.pts_instance_mask[0])
        instance_labels = tensor2ndarray(pts_seg.instance_labels)

        palette = np.array(palette)

       

        instance_indices = [np.where(instance)[0] for instance in pts_ins_mask]

        pts_ins_seg = np.zeros((points.shape[0], ), dtype=np.uint8)

        for instance_id, instance in reversed(list(enumerate(instance_indices))):
            pts_ins_seg[instance] = instance_id

        pts_color = palette[pts_ins_seg]
        seg_color = np.concatenate([points[:, :3], pts_color], axis=1)

        self.set_points(points, pcd_mode=2, vis_mode='add')
        self.draw_seg_mask(seg_color)

    def _draw_pts_sem_seg(self,
                          points: Union[Tensor, np.ndarray],
                          pts_seg: PointData,
                          palette: Optional[List[tuple]] = None,
                          ignore_index: Optional[int] = None) -> None:
        """Draw 3D semantic mask of GT or prediction.

        Args:
            points (Tensor or np.ndarray): The input point cloud to draw.
            pts_seg (:obj:`PointData`): Data structure for pixel-level
                annotations or predictions.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.
            ignore_index (int, optional): Ignore category. Defaults to None.
        """
        check_type('points', points, (np.ndarray, Tensor))

        points = tensor2ndarray(points)
        pts_sem_seg = tensor2ndarray(pts_seg.pts_semantic_mask[0])
        palette = np.array(palette)

        if ignore_index is not None:
            points = points[pts_sem_seg != ignore_index]
            pts_sem_seg = pts_sem_seg[pts_sem_seg != ignore_index]

        pts_color = palette[pts_sem_seg]
        seg_color = np.concatenate([points[:, :3], pts_color], axis=1)

        self.set_points(points, pcd_mode=2, vis_mode='add')
        self.draw_seg_mask(seg_color)

    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        assert vis_task in (
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg', 
            'multi-modality_det','lidar_sem_seg'), f'got unexpected vis_task {vis_task}.'
        classes = self.dataset_meta.get('classes', None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get('palette', None)

        if self.dataset_meta.get('dataset') == 'scannet200':

            palette = [ (0,0,0) for i in range(200)]


            vis_class_palette = {
                'wall': (0, 255, 255), 
                'floor':(0, 0, 255),
                'door':(200, 200, 100),
                'ceiling': (0, 255, 0 ),
                'window': (100, 100, 255),
                'doorframe':(170, 120, 200),
            }
            vis_class_id = [classes.index(cls) for cls in vis_class_palette.keys()]
            
            for i in vis_class_id:
                palette[i] = vis_class_palette[classes[i]]

    
        
        
        ignore_index = self.dataset_meta.get('ignore_index', None)

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None



        if draw_pred and data_sample is not None:
            if 'pred_pts_seg' in data_sample and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                instalce_palette = self.get_instance_palette(
                    data_sample.pred_pts_seg.instance_labels, palette
                )


                self._draw_pts_ins_seg(data_input['points'],
                                       data_sample.pred_pts_seg, instalce_palette,
                                       ignore_index)
                
            if 'pred_pts_seg' in data_sample and vis_task == 'lidar_sem_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                
                palette = self.dataset_meta.get('palette', None)
                self._draw_pts_sem_seg(data_input['points'],
                                       data_sample.pred_pts_seg, palette,
                                       ignore_index)


        if o3d_save_path is not None:
            out_file = osp.join(
                osp.splitext(osp.dirname(o3d_save_path))[0], 
                f'{osp.splitext(osp.basename(name))[0]}.pcd'
            )
            # out_file = o3d_save_path.split('.')[0] + f'
            
            o3d.io.write_point_cloud(out_file, self.pcd)
            # out_file = o3d_save_path.split('.')[0] + f'_{name}.ply'

            # o3d.io.write_point_cloud(out_file, self.pcd)

            

            # write_obj(self.points_colors,out_filename=out_file)
    
    def get_instance_palette(self,label_classes:List[tuple],dataset_palette:List[tuple]) -> List[tuple]:
        """Get palette for visualization.

        Args:
            palette (str or List[tuple]): The palette name or the palette list.
            num_classes (int): The number of classes.

        Returns:
            List[tuple]: The palette list.
        """
        dataset_palette = [ (self._jitter(dataset_palette[cls]))for cls in label_classes]
        return dataset_palette

    @staticmethod
    def _jitter(color):
        hsv_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0 , color[2] / 255.0)
        jitter_color = [hsv_color[0], hsv_color[1], hsv_color[2]]
        jitter_color[0] += random.uniform(-0.02, 0.02)
        jitter_color[1] += random.uniform(-0.15, 0.15)
        jitter_color[2] += random.uniform(-0.15, 0.15)
        jitter_color = np.clip(jitter_color, 0, 1)
        res = colorsys.hsv_to_rgb(jitter_color[0], jitter_color[1], jitter_color[2])
        return [res[0] * 255, res[1] * 255, res[2] * 255]
    
    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                   points_color: Tuple[float] = (0.8, 0.8, 0.8),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the point cloud to draw.

        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates): 0 represents
                LiDAR, 1 represents CAMERA, 2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:

                - 'replace': Replace the existing point cloud with input point
                  cloud.
                - 'add': Add input point cloud into existing point cloud.

                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
            mode (str): Indicate type of the input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)

        # if not hasattr(self, 'o3d_vis'):
            # self.o3d_vis = self._initialize_o3d_vis(frame_cfg)

        # for now we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        # if hasattr(self, 'pcd') and vis_mode != 'add':
        #     self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        # render_option = self.o3d_vis.get_render_option()
        # if render_option is not None:
        #     render_option.point_size = points_size
        #     render_option.background_color = np.asarray([0, 0, 0])

        points = points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        # self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors
    
    @master_only
    def draw_seg_mask(self, seg_mask_colors: np.ndarray) -> None:
        """Add segmentation mask to visualizer via per-point colorization.

        Args:
            seg_mask_colors (np.ndarray): The segmentation mask with shape
                (N, 6), whose first 3 dims are point coordinates and last 3
                dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.pts_seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.pts_seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        # self.o3d_vis.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        self.set_points(seg_points, pcd_mode=2, vis_mode='add', mode='xyzrgb')


        

