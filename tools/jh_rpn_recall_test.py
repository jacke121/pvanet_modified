#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.pascal_voc import pascal_voc
from extended_utility_by_jh.pascal_Voc_Enhanced import pascal_voc_enhanced

import caffe
import argparse
import pprint
import time, os, sys
import pickle

import ipdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    parser.add_argument( '--rpn', dest='rpn_file', help='rpn file path', type=str, default=None )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    imdb = pascal_voc_enhanced( 'test' , '2007' )

    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    imdb.set_proposal_method( 'gt' )
    rpn_file = args.rpn_file
    if os.path.exists( rpn_file ):
        with open( rpn_file , 'rb' ) as fid:
            rpn_boxes = pickle.load(fid)
            print("load rpn boxes from {}".format(rpn_file))
    else:
        print("the rpn file doesn't exit")
        sys.exit()

    #imdb.evaluate_detections_different_size_box( annotation_boxes ,  "./output/JH_test" )

    print('Evaluating detections recall hahahahah')
    imdb.evaluate_rpn_without_regression_and_classification_recall( rpn_boxes )

