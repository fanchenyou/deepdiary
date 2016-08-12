import numpy as np
import code

from utils import merge_init_structs, initw, accumNpDicts

class LSTM:
  """ 
  A multimodal long short-term memory (LSTM) generator
  """
  
  @staticmethod
  def init(params, misc):

    # inputs
    image_encoding_size = params.get('image_encoding_size', 128)
    word_encoding_size = params.get('word_encoding_size', 128)
    hidden_size = params.get('hidden_size', 128)
    num_layer = params.get('num_layer', 1)
    generator = params.get('generator', 'lstm')
    vocabulary_size = len(misc['wordtoix'])
    output_size = len(misc['ixtoword']) # these should match though
    image_size = 4096 # size of CNN vectors hardcoded here

    assert image_encoding_size == word_encoding_size, 'this implementation does not support different sizes for these parameters'

    # initialize the encoder models
    model = {}
    update = []
    regularize = []
    layer = []    

    model['lnum'] = num_layer
    
    #layer 0, input from image and text
    layer_struct = {}
    layer_struct['We'] = initw(image_size, image_encoding_size) # image encoder
    layer_struct['be'] = np.zeros((1,image_encoding_size))
    layer_struct['Ws'] = initw(vocabulary_size, word_encoding_size) # word encoder
    layer.append(layer_struct)
    update.append(['We', 'be', 'Ws'])
    regularize.append(['We', 'Ws'])

  
    #for each layer, a set of parameters
    mid_size = 0
    
    for i in xrange(1,num_layer+1):
      if i==num_layer: mid_size = output_size
      else: mid_size = hidden_size;
      layer_struct = {}
      layer_struct['WLSTM'] = initw(word_encoding_size + hidden_size + 1, 4 * hidden_size)
      layer_struct['Wd'] = initw(hidden_size, mid_size) # decoder
      layer_struct['bd'] = np.zeros((1, mid_size))
      
      layer.append(layer_struct)
      update.append(['WLSTM','Wd','bd'])
      regularize.append(['WLSTM', 'Wd'])

    model['layer'] = layer
    return { 'model' : model, 'update' : update, 'regularize' : regularize }
    

        
  @staticmethod
  def forward(batch, model, params, misc, predict_mode = False):

    F = np.row_stack(x['image']['feat'] for x in batch) 
    We = model['layer'][0]['We']
    be = model['layer'][0]['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size


    wordtoix = misc['wordtoix']
    Ws = model['layer'][0]['Ws']
    num_layer = model['lnum']
    
    gen_caches = []   # gen_caches[batch_idx] = [idx, gen_cache]   gen_cache={'WLSTM'....}  
    Ys = [] # outputs
    for i,x in enumerate(batch):

      ix = [0] + [ wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix ]      
      Xs = np.row_stack( [Ws[j, :] for j in ix] )
      Xi = Xe[i,:]
      X0 = np.row_stack([Xi, Xs])
      
      tmp_cache = []
      tmp_cache.append(ix)
      for k in xrange(1,num_layer+1):
        gen_Y, gen_cache = LSTM.forward_layer(X0, model, params, k, predict_mode)
        X0 = gen_Y
        tmp_cache.append(gen_cache)
      #print len(ix), gen_Y.shape
      Ys.append(gen_Y)
      gen_caches.append(tmp_cache)


    # back up information we need for efficient backprop
    cache = {}
    if not predict_mode:
      # use in backward pass
      cache['gen_caches'] = gen_caches
      cache['Xe'] = Xe
      cache['Ws_shape'] = Ws.shape
      cache['F'] = F
      cache['num_layer'] = num_layer

    return Ys, cache
    
    
  @staticmethod
  def forward_layer(X, model, params, l, predict_mode):
    
    drop_prob_encoder = params.get('drop_prob_encoder', 0.0)
    drop_prob_decoder = params.get('drop_prob_decoder', 0.0)

    if drop_prob_encoder > 0: # if we want dropout on the encoder
      if not predict_mode: # and we are in training mode
        scale = 1.0 / (1.0 - drop_prob_encoder)
        U = (np.random.rand(*(X.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
        X *= U # drop!

    WLSTM = model['layer'][l]['WLSTM']
    n = X.shape[0]
    d = model['layer'][l]['Wd'].shape[0] # size of hidden layer
    d2 = d*2
    d3 = d*3
    Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
    Hout = np.zeros((n, d))
    IFOG = np.zeros((n, d * 4))
    IFOGf = np.zeros((n, d * 4)) # after nonlinearity
    C = np.zeros((n, d))
    for t in xrange(n):
      # set input
      prev = np.zeros(d) if t == 0 else Hout[t-1]
      Hin[t,0] = 1
      Hin[t,1:1+d] = X[t]
      Hin[t,1+d:] = prev

      # compute all gate activations. dots:
      IFOG[t] = Hin[t].dot(WLSTM)
      
      # non-linearities
      IFOGf[t,:d3] = 1.0/(1.0+np.exp(-IFOG[t,:d3])) # sigmoids; these are the gates
      IFOGf[t,d3:] = np.tanh(IFOG[t, d3:]) # tanh

      # compute the cell activation
      C[t] = IFOGf[t,:d] * IFOGf[t, d3:]
      if t > 0: C[t] += IFOGf[t,d:d2] * C[t-1]
      Hout[t] = IFOGf[t,d2:d3] * np.tanh(C[t])

    if drop_prob_decoder > 0: # if we want dropout on the decoder
      if not predict_mode: # and we are in training mode
        scale2 = 1.0 / (1.0 - drop_prob_decoder)
        U2 = (np.random.rand(*(Hout.shape)) < (1 - drop_prob_decoder)) * scale2 # generate scaled mask
        Hout *= U2 # drop!

    # decoder at the end
    Wd = model['layer'][l]['Wd']
    bd = model['layer'][l]['bd']
    # NOTE1: we are leaving out the first prediction, which was made for the image
    # and is meaningless.
    #Y = Hout[1:, :].dot(Wd) + bd 
    Y = Hout.dot(Wd) + bd 

    gen_cache = {}
    if not predict_mode:
      # we can expect to do a backward pass
      gen_cache['WLSTM'] = WLSTM
      gen_cache['Hout'] = Hout
      gen_cache['Wd'] = Wd
      gen_cache['IFOGf'] = IFOGf
      gen_cache['IFOG'] = IFOG
      gen_cache['C'] = C
      gen_cache['X'] = X
      gen_cache['Hin'] = Hin
      gen_cache['drop_prob_encoder'] = drop_prob_encoder
      gen_cache['drop_prob_decoder'] = drop_prob_decoder
      if drop_prob_encoder > 0: gen_cache['U'] = U # keep the dropout masks around for backprop
      if drop_prob_decoder > 0: gen_cache['U2'] = U2

    return Y, gen_cache



  @staticmethod
  def backward(dY, cache):
  
    Xe = cache['Xe']
    dWs = np.zeros(cache['Ws_shape'])
    gen_caches = cache['gen_caches']
    F = cache['F']
    dXe = np.zeros(Xe.shape)
    num_layer = cache['num_layer']

    # backprop each item in the batch
    
    grads = {}
    
    for i in xrange(len(gen_caches)):
      dY0 = dY[i]

      for j in reversed(xrange(1,num_layer+1)):
        #assert len(gen_caches[i]) == num_layer + 1   #ix, [cache for each layer]
        gen_cache = gen_caches[i][j] # unpack
        local_grads = LSTM.backward_layer(dY0, gen_cache, j)
        dX = local_grads['dX']
        del local_grads['dX']
        
        if j==1: #first layer
          ix = gen_caches[i][0]
          dXi = dX[0,:]
          dXs = dX[1:,:]
          # now backprop from dXs to the image vector and word vectors
          dXe[i,:] += dXi # image vector
          for n,k in enumerate(ix): # and now all the other words, n is index in list [ix], j is ix[n]
            dWs[k,:] += dXs[n,:]
        else: dY0 = dX
        
        if j not in grads: grads[j] = {}
        accumNpDicts(grads[j], local_grads) # add up the gradients wrt model parameters

    # finally backprop into the image encoder
    dWe = F.transpose().dot(dXe)
    dbe = np.sum(dXe, axis=0, keepdims = True)

    grads[0]= { 'We':dWe, 'be':dbe, 'Ws':dWs }
    return grads
    
    
    
  @staticmethod
  def backward_layer(dY, cache, l):

    Wd = cache['Wd']
    Hout = cache['Hout']
    IFOG = cache['IFOG']
    IFOGf = cache['IFOGf']
    C = cache['C']
    Hin = cache['Hin']
    WLSTM = cache['WLSTM']
    X = cache['X']
    drop_prob_encoder = cache['drop_prob_encoder']
    drop_prob_decoder = cache['drop_prob_decoder']
    n,d = Hout.shape
    d2 = d*2
    d3 = d*3
    
    # we have to add back a row of zeros, since in the forward pass
    # this information was not used. See NOTE1 above.
    #dY = np.row_stack([np.zeros(dY.shape[1]), dY])

    # backprop the decoder
    dWd = Hout.transpose().dot(dY)
    dbd = np.sum(dY, axis=0, keepdims = True)
    dHout = dY.dot(Wd.transpose())

    # backprop dropout, if it was applied
    if drop_prob_decoder > 0:
      dHout *= cache['U2']

    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros(X.shape)
    for t in reversed(xrange(n)):

      tanhCt = np.tanh(C[t]) # recompute this here
      dIFOGf[t,d2:d3] = tanhCt * dHout[t]
      dC[t] += (1-tanhCt**2) * (IFOGf[t,d2:d3] * dHout[t])
 
      if t > 0:
        dIFOGf[t,d:d2] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,d:d2] * dC[t]
      dIFOGf[t,:d] = IFOGf[t, d3:] * dC[t]
      dIFOGf[t, d3:] = IFOGf[t,:d] * dC[t]
      
      # backprop activation functions
      dIFOG[t,d3:] = (1 - IFOGf[t, d3:] ** 2) * dIFOGf[t,d3:]
      y = IFOGf[t,:d3]
      dIFOG[t,:d3] = (y*(1.0-y)) * dIFOGf[t,:d3]

      # backprop matrix multiply
      dWLSTM += np.outer(Hin[t], dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())

      # backprop the identity transforms into Hin
      dX[t] = dHin[t,1:1+d]
      if t > 0:
        dHout[t-1] += dHin[t,1+d:]

    if drop_prob_encoder > 0: # backprop encoder dropout
      dX *= cache['U']

    return { 'WLSTM': dWLSTM, 'Wd': dWd, 'bd': dbd, 'dX': dX}

 
 
  @staticmethod
  def predict(batch, model, params, **kwparams):
    F = np.row_stack(x['image']['feat'] for x in batch) 
    We = model['layer'][0]['We']
    be = model['layer'][0]['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size

    Ys = []
    for i,x in enumerate(batch):
      gen_Y = LSTM.predict_on(Xe[i, :], model, model['layer'][0]['Ws'], params, **kwparams)
      Ys.append(gen_Y)
    return Ys
    
    
  @staticmethod
  def predict_on(Xi, model, Ws, params, **kwargs):
    """ 
    Run in prediction mode with beam search. The input is the vector Xi, which 
    should be a 1-D array that contains the encoded image vector. We go from there.
    Ws should be NxD array where N is size of vocabulary + 1. So there should be exactly
    as many rows in Ws as there are outputs in the decoder Y. We are passing in Ws like
    this because we may not want it to be exactly model['Ws']. For example it could be
    fixed word vectors from somewhere else.
    """
    tanhC_version = 1
    beam_size = kwargs.get('beam_size', 1)
    fix = kwargs.get('fix',0)


    lnum = model['lnum']
    H_prev = []
    C_prev = []
    H_prev.append([])
    C_prev.append([])
    for i in xrange(1,lnum+1):
      tmpHin = np.zeros((1,model['layer'][i]['Wd'].shape[0]))
      H_prev.append(tmpHin)
      C_prev.append(tmpHin)

    hsz = params['hidden_size']
    
    # lets define a helper function that does a single LSTM tick
    def LSTMtick(x, H_prev, C_prev):
      t = 0
      H_cur = np.zeros((lnum+1, params['hidden_size']))
      C_cur = np.zeros((lnum+1, params['hidden_size']))

      assert np.sum(H_prev[0,:])==0
      assert np.sum(C_prev[0,:])==0

      # setup the input vector
      # we start from 1 for clarity
      for i in xrange(1,lnum+1):
        Wd = model['layer'][i]['Wd']
        bd = model['layer'][i]['bd']
        WLSTM = model['layer'][i]['WLSTM']
        d = Wd.shape[0]
        assert (d == params['hidden_size'])
        assert (d == hsz)
        Hin = np.zeros((1,WLSTM.shape[0])) # xt, ht-1, bias
        Hin[t,0] = 1
        Hin[t,1:1+d] = x
        Hin[t,1+d:] = H_prev[i]

        # LSTM tick forward
        IFOG = np.zeros((1, d * 4))
        IFOGf = np.zeros((1, d * 4))
        C = np.zeros((1, d))
        Hout = np.zeros((1, d))
        IFOG[t] = Hin[t].dot(WLSTM)
        IFOGf[t,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:3*d]))
        IFOGf[t,3*d:] = np.tanh(IFOG[t, 3*d:])
        C[t] = IFOGf[t,:d] * IFOGf[t, 3*d:] + IFOGf[t,d:2*d] * C_prev[i]
        if tanhC_version:
          Hout[t] = IFOGf[t,2*d:3*d] * np.tanh(C[t])
        else:
          Hout[t] = IFOGf[t,2*d:3*d] * C[t]
        Y = Hout.dot(Wd) + bd
        C_cur[i] = C[t]
        H_cur[i] = Hout[t]
        x = Y
        
      #return (Y, Hout, C) # return output, new hidden, new cell
      return (Y, H_cur, C_cur) # return output, new hidden, new cell

    # forward prop the image
    (y0, h, c) = LSTMtick(Xi, np.zeros((lnum+1, hsz)), np.zeros((lnum+1, hsz)) )
    
    # perform BEAM search. NOTE: I am not very confident in this implementation since I don't have
    # a lot of experience with these models. This implements my current understanding but I'm not
    # sure how to handle beams that predict END tokens. TODO: research this more.
    if beam_size > 1:
      # log probability, indices of words predicted in this beam so far, and the hidden and cell states
      beams = [(0.0, [], h, c)] 
      nsteps = 0
      while True:
        beam_candidates = []
        for b in beams:
          ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
          if ixprev == 1 and b[1]:
            # this beam predicted end token. Keep in the candidates but don't expand it out any more
            beam_candidates.append(b)
            continue
          (y1, h1, c1) = LSTMtick(Ws[ixprev], b[2], b[3])
          y1 = y1.ravel() # make into 1D vector
          maxy1 = np.amax(y1)
          e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
          p1 = e1 / np.sum(e1)
          y1 = np.log(1e-20 + p1) # and back to log domain
          top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
          for i in xrange(beam_size):
            wordix = top_indices[i]
            beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
        beam_candidates.sort(reverse = True) # decreasing order
        beams = beam_candidates[:beam_size] # truncate to get new beams
        nsteps += 1
        if nsteps >= 20: # bad things are probably happening, break out
          break
      # strip the intermediates
      predictions = [(b[0], b[1]) for b in beams]
    elif beam_size==1:
      # greedy inference. lets write it up independently, should be bit faster and simpler
      ixprev = 0
      nsteps = 0
      predix = []
      predlogprob = 0.0
      while True:
        (y1, h, c) = LSTMtick(Ws[ixprev], h, c)
        ixprev, ixlogprob = ymax(y1)
        predix.append(ixprev)
        predlogprob += ixlogprob
        nsteps += 1
        if ixprev == 1 or nsteps >= 20:
          break
      predictions = [(predlogprob, predix)]
    elif beam_size == -1:             #Diverse M-Best Solutions for greedy 1-beam
      predictions=[]
      num_shift = 4
      if fix==1: num_shift=1
      for shift in xrange(num_shift):     #different penalty level of 'diversity', from 0 (hardest) to 4(lightest)
        prev_pred = []   #store previously found MAP
        for i in xrange(5):
          ixprev = 0
          nsteps = 0
          predix = []
          predlogprob = 0.0
          #(y0, h, c) = LSTMtick(Xi, np.zeros(d), np.zeros(d))
          (y0, h, c) = LSTMtick(Xi, np.zeros((lnum+1, hsz)), np.zeros((lnum+1, hsz)) )

          while True:
            (y1, h, c) = LSTMtick(Ws[ixprev], h, c)
            if prev_pred:
              for j, pred in enumerate(prev_pred):
                if nsteps < len(pred):
                  if fix==0:
                    y1[0, pred[nsteps]] -= km(j,shift)
                  else:
                    y1[0, pred[nsteps]] -= km2()


            ixprev, ixlogprob = ymax(y1)
            predix.append(ixprev)
            predlogprob += ixlogprob
            nsteps += 1
            if ixprev == 1 or nsteps >= 20:
              break
          predictions.append((predlogprob, predix, shift))  #add a term to indicate which shift is used here
          prev_pred.append(predix)
    elif beam_size == -2:  #Diverse M-Best Solutions for greedy n-beam
      # log probability, indices of words predicted in this beam so far, and the hidden and cell states
      bsize = 5
      predictions=[]
      num_shift = 4
      if fix==1: num_shift=1
      for shift in xrange(num_shift):     #different penalty level of 'diversity', from 0 (hardest) to 4(lightest)
        divCand = []  #record previously found candidate sentences
        for j in xrange(3):        #do 3 loops of beam search
          beams = [(0.0, [], h, c)]     #(logp, seq=[], h, c)
          nsteps = 0
          while True:
            beam_candidates = []
            for b in beams:
              ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off, if seq=[], start from #START# token
              if ixprev == 1 and b[1]:
                # this beam predicted end token. Keep in the candidates but don't expand it out any more
                beam_candidates.append(b)
                continue
              (y1, h1, c1) = LSTMtick(Ws[ixprev], b[2], b[3])
              y1 = y1.ravel() # make into 1D vector
              
              #embed diverse penalty
              if divCand:
                for divc in divCand:
                  assert len(divc)==3 and divc[2]<j
                  if len(b[1]) < len(divc[1]):
                    if fix==0:
                      y1[divc[1][len(b[1])]] -= km(divc[2],shift)
                    else:
                      y1[divc[1][len(b[1])]] -= km2()


              maxy1 = np.amax(y1)
              e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
              p1 = e1 / np.sum(e1)
              y1 = np.log(1e-20 + p1) # and back to log domain
              top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
              for i in xrange(bsize):
                wordix = top_indices[i]
                beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
            beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:bsize] # truncate to get new beams
            nsteps += 1
            if nsteps >= 20: # bad things are probably happening, break out
              break
          for bs in beams:
            divCand.append((bs[0],bs[1],j))
              # strip the intermediates
          #predictions.append((b[0], b[1]) for b in beams)
        for divc in divCand:
          predictions.append((divc[0],divc[1],shift,divc[2]))

    return predictions

def ymax(y):
  """ simple helper function here that takes unnormalized logprobs """
  y1 = y.ravel() # make sure 1d
  maxy1 = np.amax(y1)
  e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
  p1 = e1 / np.sum(e1)
  y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
  ix = np.argmax(y1)
  return (ix, y1[ix])

#for generating km, see Diverse M-Best Solutions in Markov Random Fields
def km(x):
  return (2**(-x))

def km(x,shift):
  return (2**(-x-shift))

def km2():
    return 1



