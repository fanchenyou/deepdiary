import time
import numpy as np
from utils import randi

from get_cost import generate_cost, generate_grads
from lstm_generator import LSTM

class Solver:
  """
  solver worries about:
  - different optimization methods, updates, weight decays
  - it can also perform gradient check
  """
  def __init__(self):
    self.step_cache_ = [] # might need this


  def step(self, batch, model, params, misc):
    """ 
    perform a single batch update. Takes as input:
    - batch of data (X)
    - model (W)
    - cost function which takes batch, model
    """
    
    learning_rate = params['learning_rate']
    update = misc['update']
    grad_clip = params['grad_clip']
    solver = params['solver']
    momentum = params['momentum']
    smooth_eps = params['smooth_eps']
    decay_rate = params['decay_rate']
    lnum = model['lnum']

    # lazily make sure we initialize step cache if needed
    if len(self.step_cache_)==0:
      for i in xrange(lnum+1):
        self.step_cache_.append({})          
        for u in update[i]:
          self.step_cache_[i][u] = np.zeros(model['layer'][i][u].shape)


    #......... forward............  
    Ys, forward_cache = LSTM.forward(batch, model, params, misc, predict_mode = False)

    # compute cost and gradient of energy function
    dYs, costs = generate_cost(batch, model, params, misc, Ys)
    
    #......... backward............
    grads = LSTM.backward(dYs, forward_cache)
    cg = generate_grads(batch, model, params, misc, grads, costs)

  
    cost = cg['cost']
    grads = cg['grad']
    stats = cg['stats']

    # clip gradients if needed, simplest possible version
    # todo later: maybe implement the gradient direction conserving version
    if grad_clip > 0:
      for i in xrange(lnum+1): 
        for p in update[i]:
          #assert p in grads[i]
          #print i,p
          grads[i][p] = np.minimum(grads[i][p], grad_clip)
          grads[i][p] = np.maximum(grads[i][p], -grad_clip)

    # perform parameter update
    for i in xrange(lnum+1): 
      for p in update[i]:
        #assert p in grads[i]
        #assert p in self.step_cache_[i]
        
        if solver == 'vanilla': # vanilla sgd, optional with momentum
          if momentum > 0:
            dx = momentum * self.step_cache_[i][p] - learning_rate * grads[i][p]
            self.step_cache_[i][p] = dx
          else:
            dx = - learning_rate * grads[i][p]
            
        elif solver == 'rmsprop':
          self.step_cache_[i][p] = self.step_cache_[i][p] * decay_rate + (1.0 - decay_rate) * grads[i][p] ** 2
          dx = -(learning_rate * grads[i][p]) / np.sqrt(self.step_cache_[i][p] + smooth_eps)

        # perform the parameter update
        model['layer'][i][p] += dx


    # create output dict and return
    out = {}
    out['cost'] = cost
    out['stats'] = stats
    return out


