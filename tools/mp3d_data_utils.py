# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np


class Matterport3DData(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', scannet200=False, save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        # self.scannet200 = scannet200

        self.classes = [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "ceiling",
            "refrigerator",
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "other",
        ]
        self.cat_ids = np.array(
            [1,2,3,4,5,6,7,8,9,10,11,12,14,16,22,24,28,33,34,36,39,]
        )

        # self.classes = [
        #     'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
        #     'bookshelf', 'picture', 'counter', 'desk', 'curtain',
        #     'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
        #     'garbagebin'
        # ]
        # self.cat_ids = np.array([
        #     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
        # ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 
                              f'scans_{split}_tiling.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'mp3d_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename).astype(np.float32)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # sp_filename = osp.join(self.root_dir, 'scannet_instance_data',
            #                         f'{sample_idx}_sp_label.npy')
            # super_points = np.load(sp_filename)
            # mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            # super_points.tofile(
            #     osp.join(self.save_path, 'super_points', f'{sample_idx}.bin'))
            # info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')

            pts_instance_mask_path = osp.join(
                self.root_dir, 'mp3d_instance_data',
                f'{sample_idx}_ins_label.npy')
            pts_semantic_mask_path = osp.join(
                self.root_dir, 'mp3d_instance_data',
                f'{sample_idx}_sem_label.npy')

            pts_instance_mask = np.load(pts_instance_mask_path).astype(
                np.int64)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                np.int64)
            _, inverse_indices = np.unique(pts_instance_mask, return_inverse=True)
            pts_instance_mask = inverse_indices + 1
            mmengine.mkdir_or_exist(
                osp.join(self.save_path, 'instance_mask'))
            mmengine.mkdir_or_exist(
                osp.join(self.save_path, 'semantic_mask'))

            pts_instance_mask.tofile(
                osp.join(self.save_path, 'instance_mask',
                            f'{sample_idx}.bin'))
            pts_semantic_mask.tofile(
                osp.join(self.save_path, 'semantic_mask',
                            f'{sample_idx}.bin'))

            info['pts_instance_mask_path'] = osp.join(
                'instance_mask', f'{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join(
                'semantic_mask', f'{sample_idx}.bin')

            info["annos"] = self.get_bboxes(
                points, pts_instance_mask, pts_semantic_mask
            )

            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def get_bboxes(self, points, pts_instance_mask, pts_semantic_mask):
        """Convert instance masks to axis-aligned bounding boxes.

        Args:
            points (np.array): Scene points of shape (n, 6).
            pts_instance_mask (np.ndarray): Instance labels of shape (n,).
            pts_semantic_mask (np.ndarray): Semantic labels of shape (n,).

        Returns:
            dict: A dict containing detection infos with following keys:

                - gt_boxes_upright_depth (np.ndarray): Bounding boxes
                    of shape (n, 6)
                - class (np.ndarray): Box labels of shape (n,)
                - gt_num (int): Number of boxes.
        """
        bboxes, labels = [], []
        for i in range(1, pts_instance_mask.max() + 1):
            ids = pts_instance_mask == i
            # check if all instance points have same semantic label
            assert pts_semantic_mask[ids].min() == pts_semantic_mask[ids].max()
            cat_id = pts_semantic_mask[ids][0]
            # keep only furniture objects
            if cat_id in self.cat_ids2class:
                labels.append(self.cat_ids2class[cat_id])
            else:
                continue
            pts = points[:, :3][ids]
            min_pts = pts.min(axis=0)
            max_pts = pts.max(axis=0)
            locations = (min_pts + max_pts) / 2
            dimensions = max_pts - min_pts
            bboxes.append(np.concatenate((locations, dimensions)))
        annotation = dict()
        # follow ScanNet and SUN RGB-D keys
        annotation["gt_boxes_upright_depth"] = np.array(bboxes)
        annotation["class"] = np.array(labels)
        annotation["gt_num"] = len(labels)
        annotation["name"] = np.array(
            [
                self.label2cat[ annotation['class'][i] ]
                for i in range(annotation["gt_num"])
            ]
        ) 
        # assert (annotation["name"].shape[0]==pts_instance_mask.max())
        return annotation
