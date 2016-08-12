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
from solver import Solver
from imagernn_utils import eval_split
from lstm_generator import LSTM
from build_vocabulary import preProBuildWordVocab
from utils import mergeVoc

def main(params):
  batch_size = params['batch_size']
  dataset = params['dataset']
  word_count_threshold = params['word_count_threshold']
  num_layer = params['num_layer']
  max_epochs = params['max_epochs']
  save_model_as = params.get('save_model_as','');
  params['tanhC_version'] = 1
  
  host = socket.gethostname() # get computer hostname

  # fetch the data provider
  dp = DataProvider(dataset)

  misc = {} # stores various misc items that need to be passed around the framework

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  misc['wordtoix'], misc['ixtoword'] = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)
  
  if params.get('expand_vocabulary', ''):
    assert params['expand_vocabulary']!=dataset
    dp2 = DataProvider(params['expand_vocabulary'])
    w2i, i2w = preProBuildWordVocab(dp2.iterSentences('train'), word_count_threshold)
    print 'expand vocabulary from %d' % (len(misc['wordtoix']),) 
    mergeVoc(misc['wordtoix'], misc['ixtoword'], w2i, i2w)
    print 'to %d' % (len(misc['wordtoix']),)

  # delegate the initialization of the model to the Generator class
  init_struct = LSTM.init(params, misc)
  model, misc['update'], misc['regularize'] = (init_struct['model'], init_struct['update'], init_struct['regularize'])



  print 'model init done.'
  print 'model has layers: %d' % (model['lnum'],)
  assert (model['lnum'] == num_layer)
  total_params = 0
  for i in xrange(model['lnum']+1):
    print 'layer %d has keys: %s' % (i, model['layer'][i].keys())
    print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model['layer'][i][k].shape[0], model['layer'][i][k].shape[1]) for k in misc['update'][i])
    print 'regularizing: ' + ', '.join( '%s [%dx%d]' % (k, model['layer'][i][k].shape[0], model['layer'][i][k].shape[1]) for k in misc['regularize'][i])
    print 'number of learnable parameters total: %d' % (sum(model['layer'][i][k].shape[0] * model['layer'][i][k].shape[1] for k in misc['update'][i]), )

  if params.get('init_model_from', ''):
    print 'init model from %s' % (params['init_model_from'],)
    # load checkpoint
    checkpoint = pickle.load(open(params['init_model_from'], 'rb'))
    model = checkpoint['model'] # overwrite the model
    misc['ixtoword'] = checkpoint['ixtoword']
    misc['wordtoix'] = checkpoint['wordtoix']

  total_params = 0
  for i in xrange(model['lnum']+1):
    print 'layer %d has keys: %s' % (i, model['layer'][i].keys())
    print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model['layer'][i][k].shape[0], model['layer'][i][k].shape[1]) for k in misc['update'][i])
    print 'regularizing: ' + ', '.join( '%s [%dx%d]' % (k, model['layer'][i][k].shape[0], model['layer'][i][k].shape[1]) for k in misc['regularize'][i])
    print 'number of learnable parameters total: %d' % (sum(model['layer'][i][k].shape[0] * model['layer'][i][k].shape[1] for k in misc['update'][i]), )


  # initialize the Solver and the cost function
  solver = Solver()

  #print size
  for sp in ['train','val','test']:
    print 'Size of %s has %d/%d images/sentences' % (sp, dp.getSplitSize(sp, ofwhat = 'images'), dp.getSplitSize(sp, ofwhat = 'sentences'))
  
  
  # calculate how many iterations we need
  #num_sentences_total = dp.getSplitSize('train', ofwhat = 'sentences')
  num_image = dp.getSplitSize('train', ofwhat = 'images')
  num_iters_one_epoch = num_image / batch_size
  #max_iters = max_epochs * num_iters_one_epoch
  eval_iter = params['eval_iter']
  top_val_ppl2 = -1
  val_ppl2 = len(misc['ixtoword'])
  abort = False;

  
  for ep in xrange(max_epochs):
    if ep>=5: params['learning_rate'] *= 0.95
    it = 0
    for batch in dp.iterImageSentencePairBatch(split = 'train', max_images = -1, max_batch_size = batch_size):
      it += 1
      if abort: break
      t0 = time.time()
      #print 'batch size', len(batch)
      
      step_struct = solver.step(batch, model, params, misc)           # get LSTM work here !!!!!!!!!!!!!!!!!!!!!!!
      cost = step_struct['cost']
      dt = time.time() - t0

      # print training statistics
      train_ppl2 = step_struct['stats']['ppl2']
      #epoch = it * 1.0 / num_iters_one_epoch
      print 'Epoch %d/%d, iter %d/%d batch done in %.3fs. loss cost = %f, reg cost = %f, ppl2 = %.2f (best %.2f), lr = %f' \
            % (ep, max_epochs, it, num_iters_one_epoch, dt, cost['loss_cost'], cost['reg_cost'], \
               train_ppl2, top_val_ppl2, params['learning_rate'])

      # perform perplexity evaluation on the validation set and save a model checkpoint if it's good
      if (it % eval_iter) == 0:
        val_ppl2 = eval_split('val', dp, model, params, misc) # perform the evaluation on VAL set
        print 'validation perplexity = %f' % (val_ppl2, )
      
        # abort training if the perplexity is no good
        min_ppl_or_abort = params['min_ppl_or_abort']
        if val_ppl2 > min_ppl_or_abort and min_ppl_or_abort > 0:
          print 'aborting job because validation perplexity %f < %f' % (val_ppl2, min_ppl_or_abort)
          abort = True # abort the job

        if val_ppl2 < top_val_ppl2 or top_val_ppl2 < 0:

            top_val_ppl2 = val_ppl2
            
            checkpoint = {}
            checkpoint['it'] = it
            checkpoint['epoch'] = ep
            checkpoint['model'] = model
            checkpoint['params'] = params
            checkpoint['perplexity'] = val_ppl2
            checkpoint['wordtoix'] = misc['wordtoix']
            checkpoint['ixtoword'] = misc['ixtoword']
            
            try:
              if save_model_as != '': 
                pickle.dump(checkpoint, open(save_model_as, "wb"))
                print 'saved checkpoint in %s' % (save_model_as, )
              else:
                filename = 'chp_%s_%s_%s_%d_%.2f.p' \
                       % (dataset, host, params['fappend'], it,val_ppl2)
                filepath = os.path.join(params['checkpoint_output_directory'], filename)
                pickle.dump(checkpoint, open(filepath, "wb"))
                print 'saved checkpoint in %s' % (filepath, )
            except Exception, e: # todo be more clever here
              print 'tried to write checkpoint into %s but got error: ' % (filepat, )
              print e




if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--init_model_from', dest='init_model_from', type=str, default='', help='initialize the model parameters from some specific checkpoint?')
  parser.add_argument('--save_model_as', dest='save_model_as', type=str, default='', help='save final model checkpoint to this file')
  parser.add_argument('--expand_vocabulary', type=str, default='', help='add another vocabulary as dummy voc, for future finetuning')

  # model parameters
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use: rnn, lstm')
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=256, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=256, help='size of word encoding')
  parser.add_argument('--num_layer', dest='num_layer', type=int, default=1, help='number of layers/depth of LSTM')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='size of hidden layer in generator RNNs')

  # optimization parameters
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, help='regularization strength')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver type: vanilla/adagrad/adadelta/rmsprop')
  parser.add_argument('--momentum', dest='momentum', type=float, default=0.0, help='momentum for vanilla sgd')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')

  # evaluation parameters
  parser.add_argument('--eval_iter', dest='eval_iter', type=int, default=100, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=100, help='for faster validation performance evaluation, what batch size to use on val img/sentences?')
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
  parser.add_argument('--min_ppl_or_abort', dest='min_ppl_or_abort', type=float , default=-1, help='if validation perplexity is below this threshold the job will abort')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
