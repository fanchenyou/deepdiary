import numpy as np
from lstm_generator import LSTM



def eval_split(split, dp, model, params, misc):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  eval_batch_size = params.get('eval_batch_size',100)
  eval_max_images = params.get('eval_max_images', -1)
  

  wordtoix = misc['wordtoix']
  ixtoword = misc['ixtoword']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  logppl = 0
  logppln = 0
  nsent = 0
  for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
    Ys, gen_caches = LSTM.forward(batch, model, params, misc, predict_mode = True)

    for i,pair in enumerate(batch):
      gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]
      gtix.append(1) # we expect END token at the end
      gtix = [0] + gtix  #insert #start# token

      Y = Ys[i]
      maxes = np.amax(Y, axis=1, keepdims=True)
      e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
      P = e / np.sum(e, axis=1, keepdims=True)
      logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
      logppln += len(gtix)
      nsent += 1
      
      #for debug
      print "gtix: ", len(gtix), [ixtoword[idx] for idx in gtix]
      outputs = np.argmax(P, axis=1)
      print "pred: ", len(outputs), [ixtoword[idx] for idx in outputs]
      

  ppl2 = 2 ** (logppl / logppln) 
  print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)
  return ppl2 # return the perplexity
