# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:36:48 2020

@author: Deise
"""

import numpy as np

def classificationEvaluation(label_gtr, label_classif, number_classes):
    Nb = label_classif.shape[0]
    label_gtr = label_gtr.astype(int)
    label_classif = label_classif.astype(int)
    #print(label_gtr[0])
    #nb_class = np.amax(label_gtr)
    #nb_class = 8  # Zurich
    #nb_class = 5  # Vaihingen
    nb_class = number_classes
    
    ConfMat = np.zeros((nb_class,nb_class))
    
    for i in range(0,Nb):
        ConfMat[label_gtr[i]-1, label_classif[i]-1] = ConfMat[label_gtr[i]-1, label_classif[i]-1] + 1
    
    po = np.sum(np.diag(ConfMat))/Nb;
    pe = 0
    
    for i in range(0,nb_class):
        pe = pe+np.sum(ConfMat[:,i]*np.sum(ConfMat[i,:]))
    pe = pe/Nb**2
    
    oca = po
    kappa = (po-pe)/(1-pe)
    
    aca = 0
    perclass_CA = np.ones((nb_class))*(-1)
    for i in range(0,nb_class):
        if (np.sum(ConfMat[i,:]) > 0): # By introducing this condition, we avoid to have a division by zero when when a semantic class do not belong to the ground truth
            perclass_CA[i] = ConfMat[i,i]/np.sum(ConfMat[i,:])
            aca = aca+perclass_CA[i]
    
    #aca = np.mean(perclass_CA)
    classes = np.unique(label_gtr)
    #aca = np.sum(perclass_CA)/classes.shape[0]
    aca = aca/classes.shape[0]    
    
    
    return oca,kappa,perclass_CA,aca,ConfMat