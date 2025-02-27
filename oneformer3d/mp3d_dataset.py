from os import path as osp
import numpy as np
import random

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset
# from mmdet3d.
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class Matterport3DDataset(ScanNetSegDataset):

    # TODO change the classes to the correct ones
    # IMPORTANT: the floor and chair categories are swapped.
    METAINFO = {
        "classes": (
            "wall",  #  [174, 199, 232],
            "floor",  # [152, 223, 138],
            "cabinet",  #  [31, 119, 180],
            "bed",  # [255, 187, 120],
            "chair",  # [188, 189, 34],
            "sofa",  # [140, 86, 75],
            "table",  # [255, 152, 150],
            "door",  # [214, 39, 40],
            "window",  #  [197, 176, 213],
            "bookshelf",  #  [148, 103, 189],
            "picture",  # [196, 156, 148],
            "counter",  # [23, 190, 207],
            "desk",  # [247, 182, 210],
            "curtain",  # [219, 219, 141],
            "ceiling",  # [255, 127, 14],
            "refrigerator",  # [158, 218, 229],
            "shower curtain",  # [44, 160, 44],
            "toilet",  # [112, 128, 144],
            "sink", # [227, 119, 194],
            "bathtub",  # [82, 84, 163],
            "other",  # [0, 0, 0],
        ),
        # the valid ids of segmentation annotations
        "seg_valid_class_ids": (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            16,
            22,
            24,
            28,
            33,
            34,
            36,
            39,
        ),
        "seg_all_class_ids": tuple(range(1, 41)),
        "palette": [
            [174, 199, 232],
            [152, 223, 138],
            [31, 119, 180],
            [255, 187, 120],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [148, 103, 189],
            [196, 156, 148],
            [23, 190, 207],
            [247, 182, 210],
            [219, 219, 141],
            [255, 127, 14],
            [158, 218, 229],
            [44, 160, 44],
            [112, 128, 144],
            [227, 119, 194],
            [82, 84, 163],
            [0, 0, 0],
        ],
    }

    """We just add super_pts_path."""

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs for data sampling."""
        return np.arange(len(self)).astype(np.int32)

    # def parse_data_info(self, info: dict) -> dict:
    #     """Process the raw data info.

    #     Args:
    #         info (dict): Raw info dict.

    #     Returns:
    #         dict: Has `ann_info` in training stage. And
    #         all path has been converted to absolute path.
    #     """
    #     # info['super_pts_path'] = osp.join(
    #     #     self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

    #     info = super().parse_data_info(info)

    #     return info
