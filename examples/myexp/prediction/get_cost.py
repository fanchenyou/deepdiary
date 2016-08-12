import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import sys


def generate_cost(batch, model, params, misc, Ys):
  """ cost function, returns cost and gradients for model """
  regc = params['regc'] # regularization cost
  wordtoix = misc['wordtoix']


  # compute softmax costs for all generated sentences, and the gradients on top
  loss_cost = 0.0
  dYs = []
  logppl = 0.0
  logppln = 0
  for i,pair in enumerate(batch):
    img = pair['image']
    # ground truth indeces for this sentence we expect to see
    gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]  
    gtix.append(1) # don't forget END token must be predicted in the end!
    gtix = [0] + gtix  #insert #start# token
    # fetch the predicted probabilities, as rows
    Y = Ys[i]
    maxes = np.amax(Y, axis=1, keepdims=True)    
    e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
    P = e / np.sum(e, axis=1, keepdims=True)
    loss_cost += - np.sum(np.log(1e-20 + P[range(len(gtix)),gtix])) # note: add smoothing to not get infs
    logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
    logppln += len(gtix)
    # P size is  n x VocSize
    # lets be clever and optimize for speed here to derive the gradient in place quickly
    for iy,y in enumerate(gtix):
      P[iy,y] -= 1 # softmax derivatives are pretty simple
    dYs.append(P)
    
    #for debug, output Y 
    '''
    ixtoword = misc['ixtoword']
    print "gtix: ", len(gtix), [ixtoword[idx] for idx in gtix]
    outputs = np.argmax(P, axis=1)
    print "pred: ", len(outputs), [ixtoword[idx] for idx in outputs]
    print
    '''
    
  return dYs, {'loss_cost': loss_cost, 'logppl': logppl, 'logppln': logppln}



def generate_grads(batch, model, params, misc, grads, costs):

  lnum = model['lnum']
  regc = params['regc'] # regularization cost
  loss_cost = costs['loss_cost']
  logppl = costs['logppl']
  logppln = costs['logppln']
  
  # add L2 regularization cost and gradients
  reg_cost = 0.0
  if regc > 0:    
    for i in xrange(lnum+1):  #zero layer has we, ws, other layers have wlstm, wd
      for p in misc['regularize'][i]:
        assert p in model['layer'][i]
        assert p in grads[i]
        assert p in misc['update'][i]
        
        mat = model['layer'][i][p]
        reg_cost += 0.5 * regc * np.sum(mat * mat)
        grads[i][p] += regc * mat

  # normalize the cost and gradient by the batch size
  batch_size = len(batch)
  reg_cost /= batch_size
  loss_cost /= batch_size
  for i in xrange(lnum+1): 
    for k in grads[i]: 
      grads[i][k] /= batch_size

  # return output in json
  out = {}
  out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
  out['grad'] = grads
  out['stats'] = { 'ppl2' : 2 ** (logppl / logppln)}
  return out

