"""
Created on Wed Aug 19 03:47:54 2020

@author: manal khallaayoune
"""

import numpy as np
import GPy
from GPy.kern.src.stationary import Stationary


class ExpSin(Stationary):
    """
    Exponential of Sinus kernel: 
        Product of 1D Exponential of Sinus kernels

    .. math::

        &k(x,x')_i = \sigma^2 \prod_{j=1}^{dimension} \exp \\bigg( - \\frac{ \sin ( x_{i,j}-x_{i,j}' ) ^2}{2 \ell_j^2} \\bigg)
        
        &x,x' \in \mathcal{M}_{n,dimension}
        
        &k \in \mathcal{M}_{n,n}

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='ExpSin'):
        super(ExpSin, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
    
    # X and X2 have the same shape
    # X.ndim = X2.ndim = 2
    # X.shape[1] = X2.shape[1] = dimension of the kernel 

    def K_of_r(self, dist):             # uses dist (distance) instead of r (radial distance)
        n = dist.shape[2]               # dist's shape: (len(X), len(X2), dimension of the kernel)
        s = 0                           # initialization of the sum
        for k in range(n):
            s+= np.sin(dist[:,:,k])**2  # the sum of the squared sinus of the distances
        return self.variance * np.exp(-s/(2*self.lengthscale**2))
    
    def K(self, X, X2):                 # redefined because it uses dist instead of r
        dist = X[:,None,:]-X2[None,:,:] # distance between X and X2
        return self.K_of_r(dist)        # returns the Covariance matrix K(X,X2)

    def dK_dr(self,dist,dimX):          # returns the partial derivative of the kernel wrt dist[:,:,dimX]
        K = self.K_of_r(dist)           # K(X,X2)
        d1 = dist[:,:,dimX]             # shape of d1: (len(X),len(X2))
        return -K*np.sin(d1)*np.cos(d1)/self.lengthscale**2
    
    def dK_dX(self, X, X2, dimX):       # returns the 1st derivative of the kernel wrt X[:,dimX]
        dist = X[:,None,:]-X2[None,:,:] # distance between X and X2
        return self.dK_dr(dist,dimX)
    
    def dK_dX2(self,X,X2,dimX2):        # returns the 1st derivative of the kernel wrt X2[:,dimX2]
        return -self.dK_dX(X,X2, dimX2)
    
    def dK2_dXdX2(self, X, X2, dimX, dimX2):    # returns the 2nd derivative of the kernel wrt X[:,dimX] and X2[:,dimX2]
        dist = X[:,None,:]-X2[None,:,:] # distance between X and X2
        K = self.K_of_r(dist)           # K(X,X2)
        dK_dX = self.dK_dX(X,X2,dimX)   # 1st derivative of the kernel wrt X[:,dimX]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)   # squared inverse of the lengthscale
        l1 = lengthscale2inv[dimX]      # lengthscale of the dimX-th part of the kernel
        l2 = lengthscale2inv[dimX2]     # lengthscale of the dimX2-th part of the kernel
        d1 = dist[:,:,dimX]             # distance of the dimX-th part of the kernel
        d2 = dist[:,:,dimX2]            # distance of the dimX2-th part of the kernel
        s1 = np.sin(d1)
        c1 = np.cos(d1)
        return (dimX!=dimX2)*dK_dX*l2*np.sin(d2)*np.cos(d2) + (dimX==dimX2)*l1*(dK_dX*s1*c1+K*c1**2-K*s1**2)
    
