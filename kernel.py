# -*- coding: utf-8 -*-
from math import exp, sqrt
import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt

def Gk_image(x,t,sigma=1):
    """
    For real vectors x and t returns their image by the
    gaussian kernel
    
    Parameters
    ----------
    x : array like - 1D array
    t : array like - 1D array
    
    Returns
    -------
    y : float

    """
    y=exp(-npl.norm(x-t)**2/sigma**2)
    
    return y
#--------------------------------------------------------

def Gk_distance(x,y,sigma=1):
    """
    for real vectors x and y computes the distance
    between their images in the gaussian kernel Hilbert 
    space 

    Parameters
    ----------
    x : array like - 1D array
    y : array like - 1D array

    Returns
    -------
    d : float

    """
    d=sqrt(2*(1-Gk_image(x,y,sigma)))
    return d

#--------------------------------------------------------

def Gk_gram_matrix(x,sigma=1):
    """
    for a dataset x return the matrix A that has for 
    entries A(i,j) = the image of the i line and the j
    line by the gaussian kernel
    
    Parameters
    ----------
    x : array like - 2D array

    Returns
    -------
    y : array like - 2D array

    """
    (n,p)=x.shape
    
    y=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            l=x[i,:]
            m=x[j,:]
            y[i][j]=Gk_image(l,m,sigma)
    y=y + y.T - np.diag(y.diagonal())
    return y

#--------------------------------------------------------

class Gk_PCA:
    
    
    def __init__(self,dataset,sigma=1):
        self.sigma=sigma
        self.dataset=dataset
        self.K=np.zeros(dataset.shape)
        self.directions=[]
        self.explained_variance=[]
        
    def compute_gram_matrix(self):
        self.K=Gk_gram_matrix(self.dataset,self.sigma)
        
    def compute_directions(self):
        n=self.K.shape[0]
        w,v=npl.eig(self.K)
        order=np.argsort(-w)
        w=w[order]
        v=v[:,order]
        
        #Directions
        D=np.zeros((n,n))
        for i in range (n):
            D[:,i]=v[:,i]/sqrt(w[i])
        
        self.directions=D
        
        #Explained_variance
        for i in range (n):
            self.explained_variance.append(1/n*D[:,i].T
                @ self.K @ self.K @ D[:,i])
        
        
        
    def reduce(self):
        return self.K @ self.directions
    
    def scree_plot(self):
        x = np.arange(self.K.shape[0]) + 1
        plt.plot(x, self.explained_variance, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.show()
        
    def plot(self):
        x=self.reduce()
        plt.scatter(x[:,0],x[:,1])
        plt.xlabel('First component')
        plt.ylabel('Second component')
        plt.show()
