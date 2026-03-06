"""CS446 kyosook2@illinois.edu
"""

import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    N = x_train.shape[0]

    alpha = torch.zeros(N, requires_grad=True)

    K = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(x_train[i], x_train[j])

    y_outer = torch.outer(y_train, y_train)
    Q = y_outer * K

    for _ in range(num_iters):
        f = 0.5 * alpha @ (Q @ alpha) - torch.sum(alpha)

        f.backward()

        with torch.no_grad():
            alpha -= lr * alpha.grad

            if c is None:
                alpha.clamp_(min=0)
            else:
                alpha.clamp_(min=0, max=c)

            y_alpha = torch.dot(alpha, y_train)
            alpha -= y_alpha / torch.dot(y_train, y_train) * y_train

        alpha.grad.zero_()

    return alpha.detach()
    

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    N = x_train.shape[0]
    M = x_test.shape[0]

    support_indices = torch.where(alpha > 1e-6)[0]
    support_alphas = alpha[support_indices]

    min_index = support_indices[torch.argmin(support_alphas)]
    x_sv = x_train[min_index]

    sum_term = 0
    for j in range(N):
        sum_term += alpha[j] * y_train[j] * kernel(x_train[j], x_sv)

    b = y_train[min_index] - sum_term

    preds = torch.zeros(M)
    for i in range(M):
        val = 0
        for j in range(N):
            val += alpha[j] * y_train[j] * kernel(x_train[j], x_test[i])
        preds[i] = val + b

    return preds
