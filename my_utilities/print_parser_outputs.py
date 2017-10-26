from easydict import EasyDict as edict
from lib.cfgs import c as dcfgs
import lib.cfgs as cfgs
import argparse

def parse_args():
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument('-tf', dest='tf_vis', help='tf devices', default=None, type=str)
    parser.add_argument('-caffe', dest='caffe_vis', help='caffe devices', default=None, type=str)
    parser.add_argument('-action', dest='action', help='action', default='train', type=str)
    attrs = ['dic', 'an', 'res']
    for d in attrs:
        for i in dcfgs[d]:
            print('-' + d +'.'+i , 'dest=' + d+'DOT'+i, 'help=' + d +'.'+i, 'default=None','type=str')
            parser.add_argument('-'+d+'.'+i, dest=d+'DOT'+i, help=d+'.'+i, default=None,type=str)

    print('-'*20)

    for i in dcfgs:
        if i not in attrs:
            print('-'+i, 'dest=' + i, 'help='+i, 'default=None','type=str')
            parser.add_argument('-'+i, dest=i, help=i, default=None,type=str)

    args = parser.parse_args()

    if args.tf_vis is not None: cfgs.tf_vis = args.tf_vis
    if args.caffe_vis is not None: cfgs.caffe_vis = args.caffe_vis
    for d in attrs:
        for i in dcfgs[d]:
            att = getattr(args, d+'DOT'+i)
            if att is not None:
                if 0:
                    print(d,i, att)
                dcfgs[d][i]=type(dcfgs[d][i])(att)

    for i in dcfgs:
        if i in attrs:
            continue
        att = getattr(args, i)
        if att is not None:
            dcfgs[i]=type(dcfgs[i])(att)

    dcfgs.Action = args.action
    if args.model is not None:
        netmodel = getattr(cfgs, args.model)
        cfgs.accname = netmodel.accname
        if args.prototxt is None:
            dcfgs.prototxt = netmodel.model
        if args.weights is None:
            dcfgs.weights = netmodel.weights
    return args


if __name__ == '__main__':

    args = parse_args()
