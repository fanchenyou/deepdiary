import argparse
import numpy as np
import code
import os
import sys
import h5py
import shutil
import tempfile

from build_vocabulary import buildWordVocab
from utils import mergeVoc
from data_provider import DataProvider

import settings as st




def data_form_expand(params, wordtoix, ixtoword):

  home_dir = st.home_dir
  exp_dir = st.exp_dir
  data_dir = st.data_dir
  h5_dir = st.h5_dir


  ################
  # Read from mat and json
  ################

  dataset1 = params['dataset1']
  dataset2 = params['dataset2']

  assert dataset1=='coco'
  assert dataset2=='lifelog' or dataset2=='lifelog_uid' or dataset2=='lifelog_dt'


  # fetch the data provider
  dp1 = DataProvider(data_dir, dataset1)
  dp2 = DataProvider(data_dir, dataset2)

  if wordtoix is None:
      assert ixtoword is None
      wordtoix, ixtoword = buildWordVocab(dp1.iterSentences('train'), params['word_count_threshold'])
      w2i, i2w = buildWordVocab(dp2.iterSentences('train'), 50)
      print '========= expand vocabulary from %d' % (len(wordtoix),) 
      mergeVoc(wordtoix, ixtoword, w2i, i2w)
      print '========= to %d' % (len(wordtoix),)
  else:
      print '========= Vocabulary Exists in data/ folder'
      

    
  #print size
  print 'dataset %s' % (dataset1,)
  for sp in ['train','val','test']:
    print 'Size of %s has %d/%d images/sentences' % (sp, dp1.getSplitSize(sp, ofwhat = 'images'), dp1.getSplitSize(sp, ofwhat = 'sentences'))
  print 'dataset %s' % (dataset2,)
  for sp in ['train','val','test']:
    print 'Size of %s has %d/%d images/sentences' % (sp, dp2.getSplitSize(sp, ofwhat = 'images'), dp2.getSplitSize(sp, ofwhat = 'sentences'))

  

  for dataset,dp in zip([dataset1, dataset2], [dp1, dp2]):
  
    ################
    # Write network definition lstm_train.prototxt
    ################

    train_h5 = os.path.join(h5_dir,'train_%s.h5' % (dataset,))
    test_h5 = os.path.join(h5_dir,'test_%s.h5' % (dataset,))
    
    names1 = [train_h5, test_h5]
  
    ################
    # Write hdf5 train/test files
    ################
  
    '''
    write input
    '''
    print 'Preparing HDF5 input files'
   
    max_y = params.get('max_len',25)

    if os.path.isfile(train_h5) and os.path.isfile(test_h5):
      print train_h5, test_h5, 'Train/test H5 files already exist'
    else:

      for ix, sp in enumerate(['train', 'val']):

        max_x = dp.getSplitSize(sp, ofwhat = 'sentences')
        idx = 0
    
        ds = 0
        nprocessed = 0
  
        X1 = np.zeros((max_x,4096)) 
        X2 = np.ones((max_x,max_y)) 
        Y =  np.ones((max_x, max_y+1)) 
  
        for out in dp.iterImageSentencePair(split = sp, max_images = -1):
          #if idx % 1000 == 0: print idx
    
          x1 = np.array(out['image']['feat'])

          tmp = [ wordtoix[w] for w in out['sentence']['tokens'] if w in wordtoix ]  
          #print tmp
          x2 = [0] + tmp
          y = [0] + tmp + [1]
          if len(y) <= 5 or len(y) > max_y:
            ds += 1
            continue
    
          #print [ixtoword[y0] for y0 in y if y0 in ixtoword]
          #print max_y, X2.shape
          #print x2
          #print y
          #print 
        
          x2 = np.array(x2)
          assert x2.shape[0] < max_y
          y = np.array(y)
    
          X1[idx] = x1          
          X2[idx, :x2.shape[0]] = x2
          Y[idx, :y.shape[0]] = y
      
          idx+=1
  
        X1 = X1[:idx,:]
        X2 = X2[:idx,:]
        Y = Y[:idx,:]
      
        #print X2
        #print Y

        print idx, X1.shape, X2.shape, Y.shape
        #solver = caffe.SGDSolver('examples/myexp/lstm_solver.prototxt')
        print 'Dropped/Remained/Total %d/%d/%d' % (ds, idx, max_x)

        with h5py.File(names1[ix], 'w') as f:
          f['img'] = X1
          f['sent'] = X2
          f['pred'] = Y


 
  train_filename = os.path.join(h5_dir,'train_%s.txt' % (dataset1,))
  test_filename = os.path.join(h5_dir,'test_%s.txt' % (dataset1,))
  with open(train_filename,'w') as f:
    f.write('%s\n' % (os.path.join(h5_dir,'train_%s.h5' % (dataset1,)),))
  with open(test_filename,'w') as f:
    f.write('%s\n' % (os.path.join(h5_dir,'test_%s.h5' % (dataset1,)),))


  train_filename = os.path.join(h5_dir,'train_%s.txt' % (dataset2,))
  test_filename = os.path.join(h5_dir,'test_%s.txt' % (dataset2,))
  with open(train_filename,'w') as f:
    f.write('%s\n' % (os.path.join(h5_dir,'train_%s.h5' % (dataset2,)),))
  with open(test_filename,'w') as f:
    f.write('%s\n' % (os.path.join(h5_dir,'test_%s.h5' % (dataset2,)),))
      
      
  
  return wordtoix, ixtoword
  