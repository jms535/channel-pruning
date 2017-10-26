from __future__ import print_function
from easydict import EasyDict as edict
from lib.cfgs import c as dcfgs
import lib.cfgs as cfgs
import os
os.environ['JOBLIB_TEMP_FOLDER']=dcfgs.shm
import argparse
os.environ['GLOG_minloglevel'] = '3'
import os.path as osp
import pickle
import sys
from multiprocessing import Process, Queue

#import google.protobuftext_format
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from lib.decompose import *
from lib.net import Net, load_layer, caffe_test
from lib.utils import *
from lib.worker import Worker

import google.protobuf.text_format

sys.path.insert(0, osp.dirname(__file__)+'/lib')


def c3(pt=cfgs.vgg.model,model=cfgs.vgg.weights):
    dcfgs.splitconvrelu=True
    cfgs.accname='accuracy@5'
    
def parse_args():
    print("@@@@ called parse_args() @@@")
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument('-tf', dest='tf_vis', help='tf devices', default=None, type=str)
    parser.add_argument('-caffe', dest='caffe_vis', help='caffe devices', default=None, type=str)
    parser.add_argument('-action', dest='action', help='action', default='train', type=str)
    attrs = ['dic', 'an', 'res']
    #print(">>> 1st for-loop")
    for d in attrs:
        for i in dcfgs[d]:
            parser.add_argument('-'+d+'.'+i, dest=d+'DOT'+i, help=d+'.'+i, default=None,type=str)
    #print(">>> 2nd for-loop")        
    for i in dcfgs:
        if i not in attrs:
            parser.add_argument('-'+i, dest=i, help=i, default=None,type=str)


    args = parser.parse_args()
    if args.tf_vis is not None: cfgs.tf_vis = args.tf_vis
    if args.caffe_vis is not None: cfgs.caffe_vis = args.caffe_vis
    #print(">>> 3rd for-loop")
    for d in attrs:
        print(d)
        for i in dcfgs[d]:
            #print(".", end="")
            att = getattr(args, d+'DOT'+i)
            if att is not None:
                print(">>>>>>",d,i, att)
                dcfgs[d][i]=type(dcfgs[d][i])(att)

    #print(">>> 4th for-loop")
    for i in dcfgs:
        if i in attrs:
            continue
        att = getattr(args, i)
        if att is not None:
            dcfgs[i]=type(dcfgs[i])(att)
    #print(">>> Last actions")
    dcfgs.Action = args.action
    if args.model is not None:
        netmodel = getattr(cfgs, args.model)
        cfgs.accname = netmodel.accname
        if args.prototxt is None:
            dcfgs.prototxt = netmodel.model
        if args.weights is None:
            dcfgs.weights = netmodel.weights
    return args


def print_dict(d, indent = 0, dic_name = "dictionary"):
    #print("@@@ print_dict() @@@")
    print(" ---%s--- " % dic_name)
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):

            print_dict(value, indent+1, str(value))
        else:
            print('\t' * (indent+1) + str(value))

if __name__ == '__main__':


    args = parse_args()
    cfgs.set_nBatches(dcfgs.nBatches)

    dcfgs.dic.option=1

    if args.action == cfgs.Action.c3:
        c3()
    else:
        pass

    print_dict(dcfgs, dic_name="dcfgs")
