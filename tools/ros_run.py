from typing import Optional, Sequence, Union, List
import logging
import os
from argparse import ArgumentParser

import numpy as np
import rospy

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array,merge_rgb_fields,array_to_pointcloud2
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from room_segmentation.msg import SegmentationResult,Instance
from mmengine.logging import print_log

from oneformer3d import OneformerSeg3DInferencer


seg_result_pub = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", help="Config file")
    parser.add_argument("weights", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--pred-score-thr", type=float, default=0.3, help="bbox score threshold"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show online visualization results"
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=-1,
        help="The interval of show (s). Demo will be blocked in showing"
        "results, if wait_time is -1. Defaults to -1.",
    )

    parser.add_argument(
        "--vis-task",
        type=str,
        default="lidar_seg",
        choices=[
            "mono_det",
            "multi-view_det",
            "lidar_det",
            "lidar_seg",
            "multi-modality_det",
            "lidar_sem_seg",
        ],
        help="Determine the visualization method depending on the task.",
    )

    call_args = vars(parser.parse_args())

    init_kws = ["model", "weights", "device"]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get("DISPLAY") is None and call_args["show"]:
        print_log(
            "Display device not found. `--show` is forced to False",
            logger="current",
            level=logging.WARNING,
        )
        call_args["show"] = False
    
    call_args["return_vis"] = True
    call_args["img_out_dir"] = ''

    return init_args, call_args


def point_cloud_cb(msg: PointCloud2, args):
    # Convert PointCloud2 message to numpy array
    inferencer: OneformerSeg3DInferencer = args["inferencer"]
    res_publisher: rospy.Publisher = args["seg_result_pub"]
    vis_publisher: rospy.Publisher = args["seg_vis_pub"]

    pc_data = pointcloud2_to_xyz_array(msg)
    # extend pcdata to 6 dim, padding with 0
    point_count = pc_data.shape[0] 
    points = np.concatenate((pc_data, np.zeros((point_count, 3))), axis=1)
    result: dict = inferencer.predict(dict(points=points), return_datasamples=False, **args["call_arg"])
    prediction = result["predictions"][0]
    visualization = result["visualization"][0] * 255

    vis_array = np.zeros(visualization.shape[0],
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), 
               ("r", np.uint32), ("g", np.uint32), ("b", np.uint32)]
    )
    vis_array["x"] = points[:, 0]
    vis_array["y"] = points[:, 1]
    vis_array["z"] = points[:, 2]
    vis_array["r"] = visualization[:, 0].astype(np.uint32)
    vis_array["g"] = visualization[:, 1].astype(np.uint32)
    vis_array["b"] = visualization[:, 2].astype(np.uint32)
    vis_array = merge_rgb_fields(vis_array)
    vis_msg = array_to_pointcloud2(vis_array,stamp=rospy.Time.now(), frame_id=msg.header.frame_id)  # Updated to use vis_array

    res_msg = SegmentationResult()
    res_msg.header.frame_id = msg.header.frame_id
    res_msg.header.stamp = rospy.Time.now()
    res_msg.header.seq = msg.header.seq

    instance_mask: np.ndarray = prediction["pts_instance_mask"]
    instance_num = instance_mask.shape[0]

    res_msg.instance_num = instance_num
    dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)]

    for i in range(instance_num):
        instance = Instance()
        instance.instance_id = i
        instance.instance_label = prediction["labels_3d"][i]
        instance.score = prediction["scores_3d"][i]
        mask_pts = pc_data[instance_mask[i] == True]
        res_ins_pt = np.zeros(mask_pts.shape[0],dtype=dtype) 
        res_ins_pt["x"] = mask_pts[:, 0]
        res_ins_pt["y"] = mask_pts[:, 1]
        res_ins_pt["z"] = mask_pts[:, 2]
        instance.instance_mask = array_to_pointcloud2(res_ins_pt)
        res_msg.instances.append(instance)
    # return pc_data
    res_publisher.publish(res_msg)
    vis_publisher.publish(vis_msg)


PointsType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def main():
    init_args, call_args = parse_args()

    inferencer = OneformerSeg3DInferencer(**init_args)

    rospy.init_node('point_cloud_subscriber', anonymous=True)
    seg_result_pub = rospy.Publisher("/seg_result", SegmentationResult, queue_size=10)
    seg_vis_pub = rospy.Publisher("/seg_vis", PointCloud2, queue_size=10)

    point_cloud_sub = rospy.Subscriber(
        "/to_semantic_cloud",
        PointCloud2,
        point_cloud_cb,
        callback_args={
            "inferencer": inferencer,
            "seg_result_pub": seg_result_pub,
            "seg_vis_pub": seg_vis_pub,
            "call_arg": call_args
        },
        queue_size=5,
    )

    rospy.spin()


if __name__ == "__main__":
    main()
