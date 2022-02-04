# -*- coding: utf-8 -*-
"""
@author: Deise Santana Maia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
matplotlib.use('Agg')
import higra as hg
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from ClassificationEvaluation import classificationEvaluation
import sap
from PIL import Image
import sys
import time
from scipy import ndimage
import tifffile as tiff
import random
from scipy.stats import rankdata
from skimage.segmentation import mark_boundaries

number_channels = 3

DATA_DIR = "/share/castor/home/santanam/Experiment_survey_article_revision/Data/Vaihingen/"
OUT_DIR = "./"

################################################################################################################

def compute_AP(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],number_channels))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],number_channels))

    for i in range(0,number_channels):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP = np.concatenate([AP_area[:,:,:,0],AP_moi[:,:,:,0]],axis=0)
    for i in range(1,number_channels):
        final_AP = np.concatenate([final_AP,AP_area[:,:,:,i],AP_moi[:,:,:,i]],axis=0)   

    return final_AP

def compute_MAX(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],number_channels))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],number_channels))

    for i in range(0,number_channels):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP_max = np.concatenate([AP_area[nb_thresholds_area:nb_thresholds_area*2+1,:,:,0],AP_moi[nb_thresholds_moi:nb_thresholds_moi*2+1,:,:,0]],axis=0)
    for i in range(1,number_channels):
        final_AP_max = np.concatenate([final_AP_max,AP_moi[nb_thresholds_moi:nb_thresholds_moi*2+1,:,:,i],AP_area[nb_thresholds_area:nb_thresholds_area*2+1,:,:,i]],axis=0)
    return final_AP_max

def compute_MIN(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],number_channels))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],number_channels))

    for i in range(0,number_channels):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP_min = np.concatenate([AP_area[0:nb_thresholds_area+1,:,:,0],AP_moi[0:nb_thresholds_moi+1,:,:,0]],axis=0)
    for i in range(1,number_channels):
        final_AP_min = np.concatenate([final_AP_min,AP_area[0:nb_thresholds_area+1,:,:,i],AP_moi[0:nb_thresholds_moi+1,:,:,i]],axis=0)

    return final_AP_min


def compute_SDAP(image, lamb_area, lamb_moi, adj):
    SDAP_area = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    SDAP_moi = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
    SDAP = sap.concatenate((SDAP_area,SDAP_moi))
    
    for i in range(1,number_channels):
        SDAP_area = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        SDAP_moi = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
        SDAP = sap.concatenate((SDAP,SDAP_area,SDAP_moi))
    
    final_SDAP = sap.vectorize(SDAP)
    
    return final_SDAP

def compute_ALPHA(image, markers, lamb_area, lamb_moi, adj):
    ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj)
    ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
   
    ALPHA_profile = sap.concatenate((ALPHA_area,ALPHA_moi))
    
    for i in range(1,number_channels):
        ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj)
        ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
        ALPHA_profile = sap.concatenate((ALPHA_profile,ALPHA_area,ALPHA_moi))
    
    final_ALPHA = sap.vectorize(ALPHA_profile)
    
    return final_ALPHA

def compute_OMEGA(image, markers, lamb_area, lamb_moi, adj):
    OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj)
    OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
    OMEGA_profile = sap.concatenate((OMEGA_area,OMEGA_moi))
    
    for i in range(1,number_channels):
        OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj)
        OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
        OMEGA_profile = sap.concatenate((OMEGA_profile,OMEGA_area,OMEGA_moi))
    
    final_OMEGA = sap.vectorize(OMEGA_profile)
    
    return final_OMEGA

def compute_WATERSHED(image, markers, lamb_area, lamb_moi,  adj, watershed_attribute):
    print("Shape of markers: ", markers.shape)
    WATERSHED_area = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj, watershed_attribute=watershed_attribute)   
    WATERSHED_moi = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,0]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max', watershed_attribute=watershed_attribute)
    WATERSHED_profile = sap.concatenate((WATERSHED_area,WATERSHED_moi))

    for i in range(1,number_channels):
        WATERSHED_area = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'area': lamb_area}, adjacency=adj, watershed_attribute=watershed_attribute)
        WATERSHED_moi = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,i]), np.ascontiguousarray(markers), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max', watershed_attribute=watershed_attribute)          
        
        WATERSHED_profile = sap.concatenate((WATERSHED_profile,WATERSHED_area,WATERSHED_moi))
    
    final_WATERSHED = sap.vectorize(WATERSHED_profile)

    return final_WATERSHED
    
################################################################################################################

def main():

    number_classes = 2 #unique.shape[0]
    nb_iter = 10
    
    # Read user parameters
    method = sys.argv[1]
    adj = int(sys.argv[2])
    watershed_attribute = ""
    bool_markers = False
    
    if (len(sys.argv) >= 4):
        watershed_attribute = sys.argv[3]
    
    if (len(sys.argv) >= 5):
        if (sys.argv[4] == "with_markers"):
            bool_markers = True

    padding_size = 2
    window_size = padding_size+1+padding_size

    #Following the training settings of [1], the images with ID = 11,15,28,30,34 used selected for test.
    #
    #[1] Yuansheng Hua et al. Semantic Segmentation of Remote Sensing Images with Sparse Annotations.
    #    IEEE Geoscience and Remote Sensing Letters. 2021. 
    
    train_indices = [1,3,5,7,13,17,21,23,26,32,37]
    test_indices = [11,15,28,30,34]
    array_colors = [(1,1,1,1),(0,0,1,1),(0,1,1,1),(0,1,0,1),(1,1,0,1),(1,0,0,1)]
    
    final_OCA = np.zeros((nb_iter))
    final_kappa = np.zeros((nb_iter))
    final_ACA = np.zeros((nb_iter))
    final_perclass = np.zeros((nb_iter, number_classes))
    final_meanIOU = np.zeros((nb_iter))

    per_class_precision = np.zeros((nb_iter,number_classes))
    per_class_recall = np.zeros((nb_iter,number_classes))
    per_class_f1 = np.zeros((nb_iter,number_classes))   
                       
    OCA = np.zeros((38,nb_iter))
    kappa = np.zeros((38,nb_iter))
    ACA = np.zeros((38,nb_iter))
    perclass = np.zeros((38,nb_iter, number_classes))

    for iteration in range(0,nb_iter):
        print("Iteration ", iteration)
        random_vaihingen_training_samples = {}
        vaihingen_training_images = {}
        vaihingen_training_markers = {}
        nb_samples_train = 0   
        
        for i in train_indices:
          vaihingen_tif = tiff.imread(DATA_DIR+'/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area'+str(i)+'.tif')
          vaihingen = np.asarray(vaihingen_tif)[:,:,0:number_channels]   
          d1 = vaihingen.shape[0]
          d2 = vaihingen.shape[1]
          vaihingen_training_markers[i] = np.ones((d1,d2)) 
          vaihingen_training_images[i] = vaihingen   
          
          
          # Select training set from one of the images of vaihingen
          img_gt_aux = tiff.imread(DATA_DIR+'/ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area'+str(i)+'.tif')
          img_gt_impervious = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
          img_gt_buildings = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 255)
          img_gt_low_vegetation = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
          img_gt_tree = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
          img_gt_car = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
          #img_gt_background = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 0)
      
          img_gt =  img_gt_impervious + 2*img_gt_buildings + 3*img_gt_low_vegetation + 4*img_gt_tree + 5*img_gt_car #+ 6*img_gt_background
    
          # 2 for buildings and 1 for the rest
          img_gt =  np.ones((d1,d2)) + 1*img_gt_buildings
          
          ################################################################
          # Random subsampling for training from the training images
          
          ################################################################
          print("Shape of training image ", str(i), ' ',d1,d2)
          random_vaihingen_training = np.zeros((d1,d2))
      
          for j in range(1,number_classes+1):
              aux = np.reshape(img_gt,(d1*d2))
              class_ = np.where(aux == j)
              if (len(class_[0]) == 0):
                  continue
              nb_samples = int(0.01*len(class_[0]))
              selected_indices = random.sample(range(len(class_[0])), nb_samples)
              class_selected_indices = np.zeros((d1*d2))
              t = list(class_[0][k] for k in selected_indices)
              class_selected_indices[t] = 1
              class_selected_indices = np.reshape(class_selected_indices,(d1,d2))
              random_vaihingen_training = random_vaihingen_training + class_selected_indices
              unique,counts = np.unique(random_vaihingen_training,return_counts=True)
              print("-----", counts, unique)
      
          nb_samples_train = nb_samples_train + int(np.sum(random_vaihingen_training))
          random_vaihingen_training = random_vaihingen_training*img_gt
          unique,counts = np.unique(random_vaihingen_training,return_counts=True)
          print("-----", counts, unique)
          random_vaihingen_training_samples[i] = random_vaihingen_training
       
        
        nb_samples_train = 0
        for i in train_indices:
            d1,d2= random_vaihingen_training_samples[i].shape
            nb_samples_train += np.sum((random_vaihingen_training_samples[i]>0)*1)
        print("Number of training samples ", nb_samples_train)


        ##########################################################################################
        # Read the test images
        
        vaihingen_test_gt = {}
        vaihingen_test_images = {}
        vaihingen_test_markers = {}
        nb_samples_test = 0    
        for i in test_indices:
          vaihingen_tif = tiff.imread(DATA_DIR+'/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area'+str(i)+'.tif')
          vaihingen = np.asarray(vaihingen_tif)[:,:,0:number_channels]   
          d1 = vaihingen.shape[0]
          d2 = vaihingen.shape[1]
          vaihingen_test_markers[i] = np.ones((d1,d2)) 
          vaihingen_test_images[i] = vaihingen   
          
          # Read test ground truth
          img_gt_aux = tiff.imread(DATA_DIR+'/ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area'+str(i)+'.tif')
          img_gt_impervious = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
          img_gt_buildings = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 255)
          img_gt_low_vegetation = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
          img_gt_tree = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
          img_gt_car = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
          #img_gt_background = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 0)
      
          img_gt =  img_gt_impervious + 2*img_gt_buildings + 3*img_gt_low_vegetation + 4*img_gt_tree + 5*img_gt_car #+ 6*img_gt_background
          
          # 2 for buildings and 1 for the rest
          img_gt =  np.ones((d1,d2)) + 1*img_gt_buildings             
            
          vaihingen_test_gt[i] = img_gt
          nb_samples_test += np.sum(img_gt>0)
    
        print("Number of test samples: ", nb_samples_test)
        
        #################################################################################################################################################
        ################################################################################################################
        
        # Area and Moment of Inertia (moi) attributes
        lamb_area= [25,100,500,1000,5000,10000,20000,50000,100000,150000]
        lamb_moi = [0.2, 0.3, 0.4, 0.5]

        ##########################################################################################
        # Extract training samples to compute the markers
        
        X_train = np.zeros((nb_samples_train,number_channels*window_size*window_size))
        y_train = np.zeros((nb_samples_train))
    
        k = 0    
        for i in train_indices:
          d1_train = vaihingen_training_images[i].shape[0]
          d2_train = vaihingen_training_images[i].shape[1] 
        
          vaihingen = vaihingen_training_images[i]
          vaihingen_padding = np.pad(vaihingen, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
          random_vaihingen_training_i = random_vaihingen_training_samples[i]
        
          for line in range(padding_size,d1_train+padding_size):
              for column in range(padding_size,d2_train+padding_size):
                  if (random_vaihingen_training_i[line-padding_size,column-padding_size] != 0):
                      X_train[k, :] = np.reshape(vaihingen_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                      y_train[k] = random_vaihingen_training_i[line-padding_size,column-padding_size]
                      k = k + 1

        
        unique,counts = np.unique(y_train,return_counts=True)
        print("Training samples: ", counts, unique)
 
        ################################################################################################################
        
        # Create the marker based on the results of the RF applied to the raw data (for the two training images and for the test image)
        
        if (bool_markers):
                 
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
                 
            # Apply to training images
            for i in vaihingen_training_images.keys():
                vaihingen = vaihingen_training_images[i]
                d1 = vaihingen.shape[0]
                d2 = vaihingen.shape[1]
                X_i = np.zeros((d1*d2,number_channels*window_size*window_size))
                vaihingen_padding = np.pad(vaihingen, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect') 
                k=0
                for line in range(padding_size,d1+padding_size):
                    for column in range(padding_size,d2+padding_size):
                        X_i[k, :] = np.reshape(vaihingen_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                        k = k + 1
                
                markers_ = clf.predict_proba(X_i)
                markers_ = np.sqrt(np.sum(markers_*markers_,axis=1))
                markers_ = 1 - ((markers_ - np.min(markers_))/(np.max(markers_) - np.min(markers_)))
          
                
                vaihingen_training_markers[i] = np.reshape(markers_, (d1, d2))
                plt.imsave(OUT_DIR+"markers_"+"_"+str(i)+"_window_"+str(window_size)+".jpg",vaihingen_training_markers[i],cmap='gray')

            # Apply to test images
            for i in vaihingen_test_images.keys():
                vaihingen = vaihingen_test_images[i]
                d1 = vaihingen.shape[0]
                d2 = vaihingen.shape[1]
                X_i = np.zeros((d1*d2,number_channels*window_size*window_size))
                vaihingen_padding = np.pad(vaihingen, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect') 
                k=0
                for line in range(padding_size,d1+padding_size):
                    for column in range(padding_size,d2+padding_size):
                        X_i[k, :] = np.reshape(vaihingen_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                        k = k + 1
                        
                markers_ = clf.predict_proba(X_i)
                markers_ = np.sqrt(np.sum(markers_*markers_,axis=1))
                markers_ = 1 - ((markers_ - np.min(markers_))/(np.max(markers_) - np.min(markers_)))
        
                      
                vaihingen_test_markers[i] = np.reshape(markers_, (d1, d2))
                plt.imsave(OUT_DIR+"markers_"+str(i)+"_window_"+str(window_size)+".png",vaihingen_test_markers[i],cmap='gray')
                
        ################################################################################################################
        
        start_time=time.time()
        all_features = {}
        
        if   (method == "GRAY"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = np.array([vaihingen[:,:,0],vaihingen[:,:,1],vaihingen[:,:,2]])
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = np.array([vaihingen[:,:,0],vaihingen[:,:,1],vaihingen[:,:,2]])
                
        elif   (method == "GRAY_5x5"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                training_gt = random_vaihingen_training_samples[i]
                d1_train = vaihingen.shape[0]
                d2_train = vaihingen.shape[1] 
                padding_size = 2
                window = padding_size*2+1
                                
                vaihingen_padding = np.pad(vaihingen, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
              
                X_train = np.zeros((window*window*number_channels, d1_train, d2_train))
                for line in range(padding_size,d1_train+padding_size):
                    for column in range(padding_size,d2_train+padding_size):
                        if (training_gt[line-padding_size, column-padding_size] > 0):
                            X_train[:, line-padding_size, column-padding_size] = np.reshape(vaihingen_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                all_features[i] = X_train
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                d1_train = vaihingen.shape[0]
                d2_train = vaihingen.shape[1] 
                padding_size = 2
                window = padding_size*2+1
                                
                vaihingen_padding = np.pad(vaihingen, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
              
                X_test = np.zeros((window*window*number_channels, d1_train, d2_train))
                for line in range(padding_size,d1_train+padding_size):
                    for column in range(padding_size,d2_train+padding_size):
                        X_test[:, line-padding_size, column-padding_size] = np.reshape(vaihingen_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                all_features[i] = X_test                          
                   
                      
        elif (method == "AP"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_AP(vaihingen, lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_AP(vaihingen, lamb_area, lamb_moi, adj)   
        elif (method == "MAX"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_MAX(vaihingen, lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_MAX(vaihingen, lamb_area, lamb_moi, adj)   
        elif (method == "MIN"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_MIN(vaihingen, lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_MIN(vaihingen, lamb_area, lamb_moi, adj)   
        elif (method == "SDAP"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_SDAP(vaihingen, lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_SDAP(vaihingen, lamb_area, lamb_moi, adj)   
        elif (method == "ALPHA"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_ALPHA(vaihingen, vaihingen_training_markers[i], lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_ALPHA(vaihingen, vaihingen_test_markers[i], lamb_area, lamb_moi, adj)  
        elif (method == "OMEGA"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_OMEGA(vaihingen, vaihingen_training_markers[i], lamb_area, lamb_moi, adj)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_OMEGA(vaihingen, vaihingen_test_markers[i], lamb_area, lamb_moi, adj)  
        elif (method == "WATERSHED"):
            for i in vaihingen_training_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_training_images[i]
                all_features[i] = compute_WATERSHED(vaihingen, vaihingen_training_markers[i], lamb_area, lamb_moi,  adj, watershed_attribute)
            for i in vaihingen_test_images.keys():
                print("Computing features of image ", i, "...")
                vaihingen = vaihingen_test_images[i]
                all_features[i] = compute_WATERSHED(vaihingen, vaihingen_test_markers[i], lamb_area, lamb_moi,  adj, watershed_attribute)  
        else:
            print("Method not implemented!")
            return

        total_time = time.time()-start_time
        print("Total time to compute features: "+ str(total_time)+"\n")
        # Extract training features from APs
        X_train = np.zeros((nb_samples_train,all_features[1].shape[0]))
        y_train = np.zeros((nb_samples_train))
        print("Number of features: ", all_features[1].shape[0])
        k = 0
        for i in vaihingen_training_images.keys():
            training_gt = random_vaihingen_training_samples[i]
            d1,d2 = training_gt.shape[0],training_gt.shape[1]
            pos_samples = np.where(training_gt > 0)
            for j in range(len(pos_samples[0])):
                X_train[k, :] = all_features[i][:, pos_samples[0][j], pos_samples[1][j]]
                y_train[k] = random_vaihingen_training_samples[i][pos_samples[0][j], pos_samples[1][j]]
                k = k+1
          
        print(np.sum(X_train < 0))  
        print("Start training...")  
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        print("Finish training...")
        
        
        ##############################################################
    
        all_y_test = []
        all_label_classif = {}
        
        # Compute evaluation results per test image
        for i in vaihingen_test_images.keys():
            test_gt = vaihingen_test_gt[i]
            d1,d2 = test_gt.shape[0],test_gt.shape[1]
            X_test = np.zeros((np.sum(test_gt>0),all_features[i].shape[0]))
            y_test = np.zeros((np.sum(test_gt>0)))
            k = 0
            for line in range(0,d1):
                for column in range(0,d2):
                    if (test_gt[line,column] > 0):
                        X_test[k,:] = all_features[i][:,line,column]
                        y_test[k]   = test_gt[line,column]
                        k=k+1
            
            all_y_test = np.concatenate((all_y_test, y_test), axis=0)  
            unique, counts = np.unique(y_test, return_counts=True)
            print("Labels of test image ", i, " : ", unique,counts)
                       
        
            label_classif = clf.predict(X_test)
            
            
            if (iteration == 0):
                k = 0
                classification_image = np.zeros((d1, d2))
                for line in range(0,d1):
                    for column in range(0,d2):
                        if (test_gt[line,column] > 0):
                            if (label_classif[k] == 1 and y_test[k] == 1):
                                classification_image[line, column] = 0
                            elif (label_classif[k] == 2 and y_test[k] == 2):
                                classification_image[line, column] = 1
                            elif (label_classif[k] == 1 and y_test[k] == 2):
                                classification_image[line, column] = 2
                            elif (label_classif[k] == 2 and y_test[k] == 1):
                                classification_image[line, column] = 3
                            k=k+1
                array_colors = [(0,0,0,1),(1,1,1,1),(1,0,0,1),(0,1,1,1)]
                unique = np.unique(classification_image)
                array_colors_aux = [array_colors[int(j)] for j in unique]
                cmap_ = colors.ListedColormap(array_colors_aux)
                plt.imshow(np.reshape(rankdata(classification_image, method='dense'), (d1,d2))-1, cmap_)
                if (bool_markers):
                    plt.imsave(OUT_DIR+'result_'+method+'_'+str(adj)+'_'+watershed_attribute+'_with_markers_'+str(i)+'_'+str(iteration+1)+'.png',classification_image,cmap=cmap_)               
                else:
                    plt.imsave(OUT_DIR+'result_'+method+'_'+str(adj)+'_'+watershed_attribute+'_with_markers_'+str(i)+'_'+str(iteration+1)+'.png',classification_image,cmap=cmap_) 
                            
                
            
            if (i != list(vaihingen_test_images.keys())[0]):
                all_label_classif[iteration] = np.concatenate((all_label_classif[iteration], label_classif), axis=0) 
            else:
                all_label_classif[iteration] = label_classif
                
            
            OCA[i,iteration],kappa[i,iteration],perclass[i,iteration],ACA[i,iteration],ConfMat = classificationEvaluation(y_test,label_classif,number_classes)
            
        final_OCA[iteration],final_kappa[iteration],final_perclass[iteration],final_ACA[iteration],final_ConfMat = classificationEvaluation(all_y_test,all_label_classif[iteration], number_classes)
        final_meanIOU[iteration] = np.sum(np.minimum(all_y_test-1,all_label_classif[iteration]-1)>0) / np.sum(np.maximum(all_y_test-1,all_label_classif[iteration]-1)>0)
        
        ##############################################################
        # Precision and recall
        for i in range(0,number_classes):
            per_class_recall[iteration,i] = ConfMat[i,i]/np.sum(ConfMat[i,:])
            per_class_precision[iteration,i] = ConfMat[i,i]/np.sum(ConfMat[:,i])
            per_class_f1[iteration,i] = 2*per_class_precision[iteration,i]*per_class_recall[iteration,i]/(per_class_precision[iteration,i]+per_class_recall[iteration,i])
                         
        ##############################################################        

    print('\n\n')
    print("Results per image ")   
    for i in vaihingen_test_images.keys():    
        if (bool_markers):
          print("Image", str(i), " ", method, ' ', watershed_attribute, " with prior knowlegde from markers")
        else:
          print("Image", str(i), " ", method, ' ', watershed_attribute)
              
        print("OA: ", np.round(np.mean(OCA[i,:]*100),2), "+-", np.round(np.std(OCA[i,:]*100),2), ", AA", np.round(np.mean(ACA[i,:]*100),2), "+-", np.round(np.std(ACA[i,:]*100),2), ", Kappa: ", np.round(np.mean(kappa[i,:]*100),2), "+-", np.round(np.std(kappa[i,:]*100),2))

        
        print("Accuracy per class: ")

        for n in range(number_classes):
            print(np.round(np.mean(perclass[i,:,n]*100),2) , "+-",  np.round(np.std(perclass[i,:,n]*100),2), ", ",  end='')
        print()
    

    print('\n')        
    print("Final results: ")
    print("OA: ", np.round(np.mean(final_OCA*100),2), "+-", np.round(np.std(final_OCA*100),2), ", AA: ", np.round(np.mean(final_ACA*100),2), "+-", np.round(np.std(final_ACA*100),2), ", Kappa: ", np.round(np.mean(final_kappa*100),2), "+-", np.round(np.std(final_kappa*100),2))
    
    print("Accuracy per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(final_perclass[:,n]*100),2) , "+-",  np.round(np.std(final_perclass[:,n]*100),2), ", ", end='')
    print('\n') 

    print("Mean IoU: ")
    print(np.round(np.mean(final_meanIOU*100),2), "+-", np.round(np.std(final_meanIOU*100),2))
    print('\n')


    print("Precision per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(per_class_precision[:,n]*100),2) , "+-",  np.round(np.std(per_class_precision[:,n]*100),2), ", ", end='')
    print('\n') 

    print("Recall per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(per_class_recall[:,n]*100),2) , "+-",  np.round(np.std(per_class_recall[:,n]*100),2), ", ", end='')
    print('\n') 
    
    print("F1 per class: ")

    for n in range(number_classes):
        print(np.round(np.mean(per_class_f1[:,n]*100),2) , "+-",  np.round(np.std(per_class_f1[:,n]*100),2), ", ", end='')
    print('\n') 
    
if __name__ == "__main__":
    main()
    
