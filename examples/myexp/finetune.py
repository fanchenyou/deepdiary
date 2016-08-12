import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import sys
import cPickle as pickle
import h5py


import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from data_form_expand import data_form_expand
import settings as st


def def_lstm_2_layer(hdf5, hidden_size, batch_size, voc_size, grad_clip, ntop, max_len, shuffle = True):
    # logistic regression: data, matrix multiplication, and 2-class softmax loss
    n = caffe.NetSpec()
    
    n.img, n.sent, n.pred = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3, shuffle=shuffle)
    
    n.input1 = L.CaptionInput(n.img, n.sent, hidden_size = hidden_size, image_size = 4096,
                    vocabulary_size = voc_size, sequence_len = max_len, weight_filler=dict(type='xavier'))

    n.drop1 = L.Dropout(n.input1, in_place=True, dropout_param=dict(dropout_ratio=0.5))

    n.lstm1 = L.Lstm(n.drop1, hidden_size = hidden_size, image_size = 4096, 
                    vocabulary_size = voc_size, sequence_len = max_len, weight_filler=dict(type='xavier'), 
                    clipping_threshold = grad_clip)
                    
    n.drop2 = L.Dropout(n.lstm1, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    
    n.ip1 = L.InnerProduct(n.drop2, num_output=hidden_size, weight_filler=dict(type='xavier'))
    
    n.lstm2 = L.Lstm(n.ip1, hidden_size = hidden_size, image_size = 4096, 
                    vocabulary_size = voc_size, sequence_len = max_len, weight_filler=dict(type='xavier'), 
                    clipping_threshold = grad_clip)
                    
    n.drop3 = L.Dropout(n.lstm2, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    
    n.ip2 = L.InnerProduct(n.drop3, num_output=voc_size, weight_filler=dict(type='xavier'))
    
    
    if ntop==1:
      n.loss = L.LstmLoss(n.ip2, n.pred, ntop=1, loss_param=dict(ignore_label=-1))  
    elif ntop==2:
      n.loss, n.pred_out = L.LstmLoss(n.ip2, n.pred, ntop=2, loss_param=dict(ignore_label=-1))  
    
    return n.to_proto()
    
    
    
def def_lstm(hdf5, hidden_size, batch_size, voc_size, grad_clip, ntop, max_len, shuffle = True):
    # logistic regression: data, matrix multiplication, and 2-class softmax loss
    n = caffe.NetSpec()
    
    n.img, n.sent, n.pred = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3, shuffle=shuffle)
    
    n.input1 = L.CaptionInput(n.img, n.sent, hidden_size = hidden_size, image_size = 4096,
                    vocabulary_size = voc_size, sequence_len = max_len, weight_filler=dict(type='xavier'))
             
    n.lstm1 = L.Lstm(n.input1, hidden_size = hidden_size, image_size = 4096, 
                    vocabulary_size = voc_size, sequence_len = max_len, weight_filler=dict(type='xavier'), 
                    clipping_threshold = grad_clip)
                    
    n.drop1 = L.Dropout(n.lstm1, in_place=True)
    
    n.ip1 = L.InnerProduct(n.drop1, num_output=voc_size, weight_filler=dict(type='xavier'))
    
    if ntop==1:
      n.loss = L.LstmLoss(n.ip1, n.pred, ntop=1, loss_param=dict(ignore_label=-1))  
    elif ntop==2:
      n.loss, n.pred_out = L.LstmLoss(n.ip1, n.pred, ntop=2, loss_param=dict(ignore_label=-1))  
    
    return n.to_proto()


def def_solver_prototxt( train_net_path, test_net_path, snapshot_prefix, max_iter=100000 ):
  s = caffe_pb2.SolverParameter()
  s.train_net = train_net_path
  s.test_net.append(test_net_path)
  s.test_interval = 5000  # Test after every 500 training iterations.
  s.test_iter.append(10) # Test on 100 batches each time we test.
  s.stepsize = 10000
  
  s.max_iter = max_iter     # no. of times to update the net (training iterations)
  s.type = "RMSProp"
  s.base_lr = 0.0001  # EDIT HERE to try different learning rates
  s.weight_decay = 0.0005
  s.lr_policy = "step"
  s.gamma = 0.5
  s.rms_decay = 0.98
  
  s.display=1000
  s.snapshot=50000
  s.snapshot_prefix = snapshot_prefix

  return s


def main(params):

  st.init()

  home_dir = st.home_dir
  exp_dir = st.exp_dir
  data_dir = st.data_dir
  h5_dir = st.h5_dir


  os.chdir(st.home_dir)  # to import caffe
  sys.path.insert(0, './python')
  
  dataset1 = params['dataset1']
  dataset2 = params['dataset2']

  num_layer = params['num_layer']
  batch_size = params['batch_size']
  hidden_size = params['hidden_size']
  grad_clip = params.get('grad_clip', 5)
  max_len = params['max_len']
  snapshot_dir = params['snapshot_dir']
  
  
  ################
  # define caffe layer prototxt
  # prepare hdf5
  ################
  voc_path = os.path.join(exp_dir, 'data/vocabulary.p')
  if os.path.exists(voc_path):
    vocabulary = pickle.load(open(voc_path, 'rb'))
    wordtoix = vocabulary['w2i']
    ixtoword = vocabulary['i2w']
    _, _ = data_form_expand(params, wordtoix, ixtoword)
  else:
    assert dataset2=='lifelog'
    wordtoix, ixtoword = data_form_expand(params, None, None)
    vocabulary = {'w2i':wordtoix, 'i2w':ixtoword}
    pickle.dump(vocabulary, open(voc_path, "wb"))

  vocabulary_size = len(wordtoix)
  #print wordtoix
  #print ixtoword
  print 'Vocabulary Size %d' % (vocabulary_size,)
  #output vocabulary, do once, and then won't change this

  ################
  # Write network definition lstm_train.prototxt
  ################
  
    
  deviceId = params['deviceId'].split(',')
  for dId in deviceId:
    d = int(dId)
    assert d>=0 and d<=2   
    caffe.set_device(d)
  caffe.set_mode_gpu()

  if num_layer==1:
    def_func = def_lstm
  elif num_layer==2:
    def_func = def_lstm_2_layer
    
  max_iters = {dataset1: 100000, dataset2: 200000}
  solver_prototxt_name = {dataset1: os.path.join(exp_dir, 'lstm_solver_%s_%s.prototxt' % (dataset1,dataset2)),
                          dataset2: os.path.join(exp_dir, 'lstm_solver_%s.prototxt' % (dataset2,))}
                   
  train_prototxt_name = {dataset1: os.path.join(exp_dir,'lstm_train_%s_%s.prototxt' % (dataset1,dataset2) ), 
                         dataset2: os.path.join(exp_dir,'lstm_train_%s.prototxt' % (dataset2,) )}
                               
  test_prototxt_name = {dataset1: os.path.join(exp_dir,'lstm_test_%s_%s.prototxt' % (dataset1,dataset2) ), 
                         dataset2: os.path.join(exp_dir,'lstm_test_%s.prototxt' % (dataset2,) )}

  snapshot_prefix = {dataset1: os.path.join(exp_dir, snapshot_dir, 'chp_%s' % (dataset1,)),
                  dataset2: os.path.join(exp_dir, snapshot_dir, 'chp_%s' % (dataset2,))}
                  
  snapshot_path = {dataset1: os.path.join(exp_dir, snapshot_dir, 'chp_%s_iter_%d.caffemodel' % (dataset1,max_iters[dataset1])),
                   dataset2: os.path.join(exp_dir, snapshot_dir, 'chp_%s_iter_%d.caffemodel' % (dataset2,max_iters[dataset2]))}
  
  for dataset in [dataset1, dataset2]:
    train_filename = os.path.join(h5_dir,'train_%s.txt' % (dataset,))
    test_filename = os.path.join(h5_dir,'test_%s.txt' % (dataset,))
    
    with open(train_prototxt_name[dataset], 'w') as f:
        f.write(str(def_func(train_filename, hidden_size, batch_size, vocabulary_size, grad_clip, 1, max_len)))
    with open(test_prototxt_name[dataset], 'w') as f:
        f.write(str(def_func(test_filename, hidden_size, batch_size, vocabulary_size, grad_clip, 1, max_len)))
    
    with open(solver_prototxt_name[dataset],'w') as f:
      s = def_solver_prototxt(train_prototxt_name[dataset],test_prototxt_name[dataset], snapshot_prefix[dataset], max_iters[dataset1])
      print s
      f.write(str(s))
    


  def display_net(net):

    #########################################
    # Display result
    #########################################


    print [(k, v[0].data.shape) for k, v in net.params.items()]
    print [(k, v.data.shape) for k, v in net.blobs.items()]
  
    We = net.params['input1'][0].data
    be = net.params['input1'][1].data
    Ws = net.params['input1'][2].data
    print 'input1 params'
    print We.shape, be.shape, Ws.shape
    print
  
    for l in xrange(1,num_layer+1):
      lstm_name = 'lstm%d' % (l,)
      
      Wi = net.params[lstm_name][0].data
      Wh = net.params[lstm_name][1].data
      bi = net.params[lstm_name][2].data
      print '==== lstm layer %d' % (l,)
      print 'Wi', Wi.shape
      print 'Wh', Wh.shape
      print 'bi', bi.shape
  
      ip_name = 'ip%d' % (l,)
    
      Wd = net.params[ip_name][0].data
      bd = net.params[ip_name][1].data
      print 'Wd', Wd.shape
      print 'bd', bd.shape
  
  


  def save_net(net, save_model_as):
    ########################################################
    #   Save to checkpoint file that python code could run
    ########################################################

    model = {}
    model['lnum'] = num_layer
    model['layer'] = []
    for l in xrange(num_layer+1): model['layer'].append({})
  
  
    We = net.params['input1'][0].data
    be = net.params['input1'][1].data
    Ws = net.params['input1'][2].data
    model['layer'][0]['We'] = We.transpose()
    model['layer'][0]['be'] = be.reshape((1,be.shape[0]))
    model['layer'][0]['Ws'] = Ws

    for l in xrange(1,num_layer+1):
      lstm_name = 'lstm%d' % (l,)
      Wi = net.params[lstm_name][0].data
      Wh = net.params[lstm_name][1].data
      bi = net.params[lstm_name][2].data
  
      ip_name = 'ip%d' % (l,)
      Wd = net.params[ip_name][0].data
      bd = net.params[ip_name][1].data

      bi = bi.reshape((bi.shape[0],1))
      WLSTM = np.concatenate((bi, Wi, Wh), axis = 1)
      #print WLSTM.shape
      model['layer'][l]['WLSTM'] = WLSTM.transpose()
      model['layer'][l]['Wd'] = Wd.transpose()
      model['layer'][l]['bd'] = bd.reshape((1,bd.shape[0]))
  
  
  
    checkpoint = {}
    checkpoint['it'] = 0
    checkpoint['epoch'] = 0
    checkpoint['model'] = model
    checkpoint['params'] = params
    checkpoint['perplexity'] = 0
    checkpoint['wordtoix'] = wordtoix
    checkpoint['ixtoword'] = ixtoword
  
  
    pickle.dump(checkpoint, open(save_model_as, "wb"))
    print 'saved checkpoint in %s' % (save_model_as, )
    
  

    
  
  #########################################
  # only train on 'lifelog'
  #########################################
  if params['lifelog_only'] ==1:
    ss = os.path.join(exp_dir, 'lstm_solver_only_lifelog.prototxt')
    sp = os.path.join(exp_dir, snapshot_dir, 'chp_only_lifelog' )
    
    with open(ss,'w') as f:
        s = def_solver_prototxt(train_prototxt_name['lifelog'],test_prototxt_name['lifelog'], sp, 200000)
        print s
        f.write(str(s))
    
    solver0 = caffe.get_solver(ss)
    solver0.solve() 
    save_net(solver0.net, os.path.join(exp_dir, 'models', 'caffe_lstm_NO_finetune.p'))
    del solver0
  
  else:
    print '===== No Lifelog only'
  #########################################
  # pre-train on 'coco'
  #########################################
  
  solver = caffe.get_solver(solver_prototxt_name[dataset1])
  t0 = time.time()
  solver.solve()  # train !!!!
  save_net(solver.net, os.path.join(exp_dir, 'models', 'caffe_lstm_COCO.p'))

  dt = time.time() - t0
  print 'Total training time %.3fs' % (dt,)

  
  #########################################
  # finetune on 'lifelog'
  #########################################
  #solver = caffe.get_solver(os.path.join(exp_dir, 'lstm_solver_%s.prototxt' % (dataset2,)))
  solver = caffe.get_solver(solver_prototxt_name[dataset2])
  #solver.net.copy_from(os.path.join(exp_dir,'%s/chp_%s_iter_100000.caffemodel' % (params['snapshot_dir'], dataset1)))  #get from pre-trained network
  solver.net.copy_from(snapshot_path[dataset1])
  #t0 = time.time()
  solver.solve()  # train !!!!
  dt = time.time() - t0
  print 'Total training time %.3fs' % (dt,)
  net = solver.net

    
  save_model_as = params.get('save_model_as', '')
  if save_model_as:
    save_model_as = os.path.join(exp_dir, save_model_as)
    save_net(net, save_model_as)
  
  
  # then go to blitzle, run ./predict_caffe_model.sh
  
  
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('-d1', '--dataset1', dest='dataset1', default='coco', help='dataset: flickr8k/flickr30k')
  parser.add_argument('-d2', '--dataset2', dest='dataset2', default='lifelog', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--deviceId', dest='deviceId', type=str , default="1", help='Define on which GPUs to run')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='size of hidden layer in generator RNNs')
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='batch size')
  parser.add_argument('--num_layer', dest='num_layer', type=int, default=1, help='number of layers/depth of LSTM')
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')
  parser.add_argument('--solver', dest='solver', type=str, default="sgd", help='if a word occurs less than this number of times in training data, it is discarded')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--save_model_as', dest='save_model_as', type=str, default='', help='save final model checkpoint to this file')
  parser.add_argument('--max_len', dest='max_len', type=int, default=25, help='length of words')
  parser.add_argument('--snapshot_dir', dest='snapshot_dir', type=str, default='snapshot', help='save final model checkpoint to this file')
  parser.add_argument('--lifelog_only', dest='lifelog_only', type=int, default=0, help='number of layers/depth of LSTM')

  
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
