import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io

from lstm_generator import LSTM


"""
This script is used to predict sentences for arbitrary images
that are located in a folder we call root_folder. It is assumed that
the root_folder contains:
- the raw images
- a file tasks.txt that lists the images you'd like to use
- a file vgg_feats.mat that contains the CNN features. 
  You'll need to use the Matlab script I provided and point it at the
  root folder and its tasks.txt file to save the features.

Then point this script at the folder and at a checkpoint model you'd
like to evaluate.
"""

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  #dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  wordtoix = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  # output blob which we will dump to JSON for visualizing the results
  blob = {} 
  blob['imgblobs'] = []

  # load the tasks.txt file
  root_path = params['root_path']
  img_names = open(os.path.join(root_path, 'tasks.txt'), 'r').read().splitlines()

  # load the features for all images
  features_path = os.path.join(root_path, 'vgg_feats.mat')
  features_struct = scipy.io.loadmat(features_path)
  features = features_struct['feats'] # this is a 4096 x N numpy array of features
  D,N = features.shape

  # iterate over all images and predict sentences
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    img = {}
    img['feat'] = features[:, n]
    img['local_file_path'] = os.path.join(root_path, 'imgs', img_names[n]);

    # perform the work. heavy lifting happens inside
    kwparams = { 'beam_size' : params['beam_size'], 'fix' : params['fix_diverse'] }
    Ys = LSTM.predict([{'image':img}], model, checkpoint_params, **kwparams)

    # build up the output
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']
    img_blob['candidates'] = []

    # encode the top predictions    
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    
    lenYs = len(top_predictions)
    for t in xrange(lenYs):
      top_prediction = top_predictions[t] # these are sorted with highest on top
      candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
      print 'PRED %d: (%f) %s' % (t, top_prediction[0], candidate)
      dtmp={}
      dtmp['text']=candidate
      dtmp['logprob']=top_prediction[0]
      if len(top_prediction) >= 3 : dtmp['shift']=top_prediction[2]
      if len(top_prediction) >= 4 : dtmp['loop']=top_prediction[3]
      #img_blob['candidates'].append({'text': candidate, 'logprob': top_prediction[0]})
      img_blob['candidates'].append(dtmp)
    blob['imgblobs'].append(img_blob)


  # dump output html
  html = ''
  for img in blob['imgblobs']:
    html += '<img src="%s" height="400"><br>' % (img['img_path'], )
    html += 'Image Name: %s <br>' % (os.path.basename(img['img_path']))
    #print img['img_path']
    #print os.path.basename(img['img_path'])
    html += '============================================================<br>'
    html += 'Diversity Level(0 highest),  Logprob,  Predicted sentence<br>'
    html += '============================================================<br>'

    for cand in img['candidates']:
      if 'shift' in cand and 'loop' in cand:
        html += '[%d.%d] (%f)  %s <br>' % (cand['shift'], cand['loop'], cand['logprob'], cand['text'])
      elif 'shift' in cand:
        html += '[%d] (%f)  %s <br>' % (cand['shift'], cand['logprob'], cand['text'])
      else:
        html += '(%f)  %s <br>' % (cand['logprob'], cand['text'])
  html += '<br>'
  html_file = params['result_html']
  print 'writing html result file to %s...' % (html_file, )
  open(html_file, 'w').write(html)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-r', '--root_path', default='example_images', type=str, help='folder with the images, tasks.txt file, and corresponding vgg_feats.mat file')
  parser.add_argument('-k', '--result_html', default='show_result.html', type=str, help='html page to show result')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('-f', '--fix_diverse', type=int, default=0, help='penalty level fixed or not.')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
