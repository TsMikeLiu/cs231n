from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_num= X.shape[0]
    X_dim = X.shape[1]
    Class_num = W.shape[1]

    for i in range(X_num):
        score = X[i].dot(W)
        score = score - np.max(score) # to avoid exponential explosion
        exp_score = np.exp(score)
         
        loss += (-np.log(exp_score[y[i]]/np.sum(exp_score)))
        ####################### CALC dW #######################################
        # loss += -np.log(exp_score[y[i]]) + np.log(np.sum(exp_score))
        d_exp_score = np.zeros_like(exp_score) # (C,)
        d_exp_score[y[i]] -= 1/exp_score[y[i]] / X_num
        d_exp_score += 1/np.sum(exp_score) / X_num
        d_score = d_exp_score * exp_score #(C,)
        dW += X[[i]].T.dot([d_score])
    pass
    loss /= X_num
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    X_num = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X @ W # (N,D) * (D,C)
    scores -= np.max(scores,axis=1,keepdims=True) # to avoid exponential explosion
    loss_scores = -scores[range(X_num),y] + np.log(np.sum(np.exp(scores),axis=1)) # (N,)
    loss = np.sum(loss_scores)/X_num + reg*np.sum(W**2)  
    
    # calculate dW should use the reverse order of loss calculate
    d_loss = np.ones_like(loss_scores)/X_num # (N,)
    d_scores_local = np.exp(scores) / np.sum(np.exp(scores))# (N,C) derivative of np.log(np.sum(np.exp(scores),axis=1))

    d_scores_local[range(X_num),y] -= 1
    d_scores = d_loss.reshape(X_num,1) * d_scores_local # (N,C)
    dW = X.T.dot(d_scores) # (D,N)*(N,C) = (D,C)
    dW += 2*reg*W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
