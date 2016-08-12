import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import random
import sys

def init():
  global home_dir
  global exp_dir
  global data_dir
  global h5_dir
  global pred_dir
  
  home_dir = '/home/fan6/lstm/caffe-caption'
  exp_dir = os.path.join(home_dir, 'examples/myexp')
  data_dir = os.path.join(exp_dir, 'data')
  h5_dir = os.path.join(exp_dir, 'h5_data')
  pred_dir = os.path.join(exp_dir, 'pred_data')