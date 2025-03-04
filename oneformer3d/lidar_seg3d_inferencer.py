# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from mmdet3d.registry import INFERENCERS
from mmdet3d.utils import ConfigType
from mmdet3d.apis.inferencers import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name="oneformer3d-seg3d-lidar")
@INFERENCERS.register_module()
class OneformerSeg3DInferencer(Base3DInferencer):
    """The inferencer of LiDAR-based segmentation.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointnet2-ssg_s3dis-seg" or
            "configs/pointnet2/pointnet2_ssg_2xb16-cosine-50e_s3dis-seg.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'mmdet3d'.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        "return_vis",
        "show",
        "wait_time",
        "draw_pred",
        "pred_score_thr",
        "img_out_dir",
        "no_save_vis",
        "no_save_pred",
        "out_dir",
        "save_dir",
        "vis_task",
    }
    postprocess_kwargs: set = {"print_result", "pred_out_file", "return_datasample"}

    def __init__(
        self,
        model: Union[ModelType, str, None] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        scope: str = "mmdet3d",
        palette: str = "none",
    ) -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(OneformerSeg3DInferencer, self).__init__(
            model=model, weights=weights, device=device, scope=scope, palette=palette
        )

    def _inputs_to_list(self, inputs: Union[dict, list]) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        return super()._inputs_to_list(inputs, modality_key="points")

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.eval_dataloader.dataset.pipeline
        # Load annotation is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, "LoadAnnotations3D")
        if idx != -1:
            del pipeline_cfg[idx]

        idx = self._get_transform_idx(pipeline_cfg, "LoadAnnotations3D_")
        if idx != -1:
            del pipeline_cfg[idx]
        idx = self._get_transform_idx(pipeline_cfg, "SwapChairAndFloor")
        if idx != -1:
            del pipeline_cfg[idx]

        idx = self._get_transform_idx(pipeline_cfg, "PointSegClassMapping")
        if idx != -1:
            del pipeline_cfg[idx]

        # idx = self._get_transform_idx(pipeline_cfg, 'MultiScaleFlipAug3D')
        # if idx != -1:
        #     idx2 = self._get_transform_idx(pipeline_cfg[idx]['transforms'],"AddSuperPointAnnotations")
        #     if idx2 != -1:
        #         del pipeline_cfg[idx]['transforms'][idx2]

        load_point_idx = self._get_transform_idx(pipeline_cfg, "LoadPointsFromFile")
        if load_point_idx == -1:
            raise ValueError("LoadPointsFromFile is not found in the test pipeline")

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg["coord_type"], load_cfg["load_dim"]
        self.use_dim = (
            list(range(load_cfg["use_dim"]))
            if isinstance(load_cfg["use_dim"], int)
            else load_cfg["use_dim"]
        )

        pipeline_cfg[load_point_idx]["type"] = "LidarDet3DInferencerLoader"
        return Compose(pipeline_cfg)

    def visualize(
        self,
        inputs: InputsType,
        preds: PredType,
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        draw_pred: bool = True,
        pred_score_thr: float = 0.3,
        img_out_dir: str = "",
        vis_task: str = "lidar_seg",
    ) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """

        if self.visualizer is None or (
            not show and img_out_dir == "" and not return_vis
        ):
            return None

        if getattr(self, "visualizer") is None:
            raise ValueError(
                'Visualization needs the "visualizer" term'
                "defined in the config, but got None."
            )

        results = []

        for single_input, pred in zip(inputs, preds):
            single_input = single_input["points"]
            if isinstance(single_input, str):
                pts_bytes = mmengine.fileio.get(single_input)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = osp.basename(single_input).split(".bin")[0]
                pc_name = f"{pc_name}.png"
            elif isinstance(single_input, np.ndarray):
                points = single_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f"pc_{pc_num}.png"
            else:
                raise ValueError("Unsupported input type: " f"{type(single_input)}")

            o3d_save_path = (
                osp.join(img_out_dir, pc_name) if img_out_dir != "" else None
            )
            if o3d_save_path is not None:
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))

            data_input = dict(points=points)
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                vis_task=vis_task,
            )
            results.append(self.visualizer.points_colors)
            self.num_visualized_frames += 1

        return results

    def pred2dict(self, data_sample: InstanceData) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.
        """
        result = {}
        if "pred_instances_3d" in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist(),
                'labels_3d': pred_instances_3d.labels_3d.tolist(),
                'scores_3d': pred_instances_3d.scores_3d.tolist()
            }

        if "pred_pts_seg" in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result["pts_semantic_mask"] = pred_pts_seg.pts_semantic_mask[0]
            result['pts_instance_mask'] = pred_pts_seg.pts_instance_mask[0]
            result["labels_3d"] = pred_pts_seg.instance_labels.tolist()
            result["scores_3d"] = pred_pts_seg.instance_scores.tolist()

        return result

    def predict(
        self,
        inputs: InputsType,
        return_datasamples: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        for data in inputs:
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(
            preds, visualization, return_datasamples, **postprocess_kwargs
        )

        return results
