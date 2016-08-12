import time
import datetime
import numpy as np


def preProBuildWordVocab(sentence_iterator, word_count_threshold):
  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation
  print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sent in sentence_iterator:
    nsents += 1
    for w in sent['tokens']:
      w = w.lower()
      word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

  ixtoword = {}
  ixtoword[0] = '#START#'
  ixtoword[1] = '.'  # period at the end of the sentence. make first dimension be end token

  wordtoix = {}
  wordtoix['#START#'] = 0 # start token
  wordtoix['.'] = 1

  ix = 2
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  return wordtoix, ixtoword


