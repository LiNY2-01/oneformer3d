# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

from mmengine.logging import print_log

from oneformer3d import OneformerSeg3DInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--img-out-dir',
        type=str,
        default='',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--vis-task',
        type=str,
        default='lidar_seg',
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det','lidar_sem_seg'
        ],
        help='Determine the visualization method depending on the task.'
    )

    call_args = vars(parser.parse_args())

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False
    call_args["return_vis"] = True

    return init_args, call_args


def main():
    # TODO: Support inference of point cloud numpy file.
    init_args, call_args = parse_args()

    inferencer = OneformerSeg3DInferencer(**init_args)
    inferencer(**call_args)

    if call_args['img_out_dir'] != '' :
        print_log(
            f'results have been saved at {call_args["img_out_dir"]}',
            logger='current')


if __name__ == '__main__':
    main()
