import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import sys
import cPickle as pickle

from data_provider import DataProvider
from build_vocabulary import preProBuildWordVocab
from utils import mergeVoc

def main(params):
  dataset = params['dataset']

  # fetch the data provider
  dp = DataProvider(dataset)

  misc = {} # stores various misc items that need to be passed around the framework

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  misc['wordtoix'], misc['ixtoword'] = preProBuildWordVocab(dp.iterSentences('train'), 10)
  
  if params.get('expand_vocabulary', ''):
    assert params['expand_vocabulary']!=dataset
    dp2 = DataProvider(params['expand_vocabulary'])
    w2i, i2w = preProBuildWordVocab(dp2.iterSentences('train'), 10)
    print 'expand vocabulary from %d' % (len(misc['wordtoix']),) 
    mergeVoc(misc['wordtoix'], misc['ixtoword'], w2i, i2w)
    print 'to %d' % (len(misc['wordtoix']),)



  
  #print size
  for sp in ['train','val','test']:
    print 'Size of %s has %d/%d images/sentences' % (sp, dp.getSplitSize(sp, ofwhat = 'images'), dp.getSplitSize(sp, ofwhat = 'sentences'))
  
  

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
