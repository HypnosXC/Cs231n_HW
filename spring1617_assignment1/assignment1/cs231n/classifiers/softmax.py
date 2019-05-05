import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  train_num=X.shape[0]
  train_cla=W.shape[1]
  scores=np.dot(X,W)
  scores-=np.max(scores,axis=1).reshape(train_num,1)
  scores=np.exp(scores)
  for i in range(train_num):
      basic_sco=scores[i,:]
      basic_sco=basic_sco.reshape(1,train_cla)
      tot=np.sum(basic_sco)
      basic_sco/=tot
      ent=basic_sco[:,y[i]]/np.sum(basic_sco)
      loss-=np.log(ent)
      dW[:,y[i]]+=-X[i,:]
      for j in range(train_cla):
          dW[:,j]+=X[i,:]*basic_sco[:,j]/np.sum(basic_sco)
        
  loss=loss/train_num+0.5*reg*np.sum(W*W)
  dW=dW/train_num+reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  train_num=X.shape[0]
  train_cla=W.shape[1]
  scores=np.dot(X,W)
  scores-=np.max(scores,axis=1).reshape(train_num,1)
  scores=np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(train_num,1)
  mask=np.zeros_like(scores)
  mask[np.arange(train_num),y]=1.0
  loss=-np.sum(mask*np.log(scores))
  loss=loss/train_num+0.5*reg*np.sum(W*W)
  #loss query finished
  dW-=np.dot(X.T,mask-scores)/train_num
  dW+=reg*W
  #dW finished
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

