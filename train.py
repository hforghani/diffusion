# -*- coding: utf-8 -*-
import getopt
import logging
import sys

from saito import SaitoDiffusion

logger = logging.getLogger('train')

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'i:', ['iterations='])
    except getopt.GetoptError:
        print 'train.py [-i <iterations_num>]'
        sys.exit(2)
    iterations = None
    for opt, arg in opts:
        if opt in ('-i', '--iterations'):
            iterations = int(arg)

    SaitoDiffusion().fit().train(iterations=iterations)
