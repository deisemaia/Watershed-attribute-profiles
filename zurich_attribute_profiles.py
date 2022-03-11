# -*- coding: utf-8 -*-
"""
@author: Deise Santana Maia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import higra as hg
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


from AllProfiles import *


number_channels = 4 # The experiments on the Zurich dataset are performed using the four channels (Infrared, Red, Green and Blue)
nb_iter = 1         # Number of iterations over which the evaluation is computed. The final scores are an average of the scores obtained for each iteration.

DATA_DIR = "./Data/"
OUT_DIR  = "./"
    
    
################################################################################################################

def main():
    
    # Read user parameters
    bool_only_buildings = False
    watershed_attribute = ""
    bool_prior_knowledge = False
    number_classes = 8 #unique.shape[0]    
    if (sys.argv[1] == "building_extraction"):
        bool_only_buildings = True
        number_classes = 2
        print("Computing evaluation results only for the building class...\n")     
    elif (sys.argv[1] == "land_cover"):
        print("Computing evaluation results on all eight semantic classes...\n")
        
    method = sys.argv[2]
    if (method == "WATERSHED_COMPUTED_WITH_PK" or method=="WATERSHED_FILTERED_WITH_PK"):
        bool_prior_knowledge = True
    adj = int(sys.argv[3])

    
    if (len(sys.argv) >= 5):
        watershed_attribute = sys.argv[4]

    
    ################################################################################################################
    
    #Following the training settings of [1] and [2], we selected the first 15 images as the training set.
    #
    #[1] Yanwei Cui, Laetitia Chapel and Sebastien Lefevre. Scalable Bag of Subpaths Kernel for Learning on
    #    Hierarchical Image Representations and Multi-Source Remote Sensing Data Classification. Remote Sensing. 2017.
    #
    #[2] Yuansheng Hua et al. Semantic Segmentation of Remote Sensing Images with Sparse Annotations.
    #    IEEE Geoscience and Remote Sensing Letters. 2021. 
    
    train_indices = range(1,16)
    test_indices = range(16,21)
    array_colors = [(1,1,1,1),(0,0,0,1),(0.4,0.4,0.4,1),(0,0.5,0,1),(0,1,0,1),(0.6,0.3,0,1),(0,0,0.6,1),(1,1,0,1),(0.6,0.6,1,1)]
    
    final_OCA = np.zeros((nb_iter))
    final_kappa = np.zeros((nb_iter))
    final_ACA = np.zeros((nb_iter))
    final_perclass = np.zeros((nb_iter, number_classes))
    final_meanIOU = np.zeros((nb_iter)) # Only used when we evaluate building extraction
    
    per_class_precision = np.zeros((nb_iter,number_classes))
    per_class_recall = np.zeros((nb_iter,number_classes))
    per_class_f1 = np.zeros((nb_iter,number_classes))   
    
    OCA = np.zeros((21,nb_iter))
    kappa = np.zeros((21,nb_iter))
    ACA = np.zeros((21,nb_iter))
    perclass = np.zeros((21,nb_iter, number_classes))

    for iteration in range(0,nb_iter):
        print("Iteration ", iteration)
        random_zurich_training_samples = {}
        zurich_training_images = {}
        zurich_training_markers = {}
        nb_samples_train = 0   
        
        for i in train_indices:
              zurich_tif = tiff.imread(DATA_DIR+'Zurich_dataset_v1.0/images_tif/zh'+str(i)+'.tif')
              zurich = np.asarray(zurich_tif)[:,:,0:number_channels]   
              d1 = zurich.shape[0]
              d2 = zurich.shape[1]
              zurich_training_markers[i] = np.ones((d1,d2)) 
              zurich_training_images[i] = zurich   
              
              
              # Select training set from one of the images of Zurich
              img_gt_aux = tiff.imread(DATA_DIR + 'Zurich_dataset_v1.0/groundtruth/zh'+str(i)+'_GT.tif')
              img_gt_background = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
              img_gt_roads = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 0)
              img_gt_buildings = (img_gt_aux[:,:,0] == 100) * (img_gt_aux[:,:,1] == 100) * (img_gt_aux[:,:,2] == 100)
              img_gt_trees = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 125) * (img_gt_aux[:,:,2] == 0)
              img_gt_grass = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
              img_gt_soil = (img_gt_aux[:,:,0] == 150) * (img_gt_aux[:,:,1] == 80) * (img_gt_aux[:,:,2] == 0)
              img_gt_water = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 150)
              img_gt_railways = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)     
              img_gt_pools = (img_gt_aux[:,:,0] == 150) * (img_gt_aux[:,:,1] == 150) * (img_gt_aux[:,:,2] == 255)   
          
              img_gt =  img_gt_roads + 2*img_gt_buildings + 3*img_gt_trees + 4*img_gt_grass + 5*img_gt_soil + 6*img_gt_water + 7*img_gt_railways + 8*img_gt_pools
        
              if (bool_only_buildings):
                  img_gt =  np.ones((d1,d2)) + 1*img_gt_buildings
            
              ################################################################
              # Random subsampling of training pixels          
              ################################################################
              
              print("Shape of training image ", str(i), ' ',d1,d2)
              random_zurich_training = np.zeros((d1,d2))
          
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
                  random_zurich_training = random_zurich_training + class_selected_indices
                  unique,counts = np.unique(random_zurich_training,return_counts=True)
                  print("-----", counts, unique)
          
              nb_samples_train = nb_samples_train + int(np.sum(random_zurich_training))
              random_zurich_training = random_zurich_training*img_gt
              unique,counts = np.unique(random_zurich_training,return_counts=True)
              print("-----", counts, unique)
              random_zurich_training_samples[i] = random_zurich_training
        
        
        nb_samples_train = 0
        for i in train_indices:
            nb_samples_train += np.sum((random_zurich_training_samples[i]>0)*1)
        print("Number of training samples ", nb_samples_train)
        

        ##########################################################################################
        # Read the test images
        
        zurich_test_gt = {}
        zurich_test_images = {}
        zurich_test_markers = {}
        nb_samples_test = 0    
        for i in test_indices:
          zurich_tif = tiff.imread(DATA_DIR+'Zurich_dataset_v1.0/images_tif/zh'+str(i)+'.tif')
          zurich = np.asarray(zurich_tif)[:,:,0:number_channels]   
          d1 = zurich.shape[0]
          d2 = zurich.shape[1]
          zurich_test_markers[i] = np.ones((d1,d2)) 
          zurich_test_images[i] = zurich   
          
          # Read test ground truth
          img_gt_aux = tiff.imread(DATA_DIR+'Zurich_dataset_v1.0/groundtruth/zh'+str(i)+'_GT.tif')
          img_gt_background = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 255)
          img_gt_roads = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 0)
          img_gt_buildings = (img_gt_aux[:,:,0] == 100) * (img_gt_aux[:,:,1] == 100) * (img_gt_aux[:,:,2] == 100)
          img_gt_trees = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 125) * (img_gt_aux[:,:,2] == 0)
          img_gt_grass = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)
          img_gt_soil = (img_gt_aux[:,:,0] == 150) * (img_gt_aux[:,:,1] == 80) * (img_gt_aux[:,:,2] == 0)
          img_gt_water = (img_gt_aux[:,:,0] == 0) * (img_gt_aux[:,:,1] == 0) * (img_gt_aux[:,:,2] == 150)
          img_gt_railways = (img_gt_aux[:,:,0] == 255) * (img_gt_aux[:,:,1] == 255) * (img_gt_aux[:,:,2] == 0)     
          img_gt_pools = (img_gt_aux[:,:,0] == 150) * (img_gt_aux[:,:,1] == 150) * (img_gt_aux[:,:,2] == 255)   
      
          img_gt =  img_gt_roads + 2*img_gt_buildings + 3*img_gt_trees + 4*img_gt_grass + 5*img_gt_soil + 6*img_gt_water + 7*img_gt_railways + 8*img_gt_pools      
          
          if (bool_only_buildings):
              img_gt =  np.ones((d1,d2)) + 1*img_gt_buildings
          
          zurich_test_gt[i] = img_gt
          nb_samples_test += np.sum(img_gt>0)
    
        print("Number of test samples: ", nb_samples_test)
        
        #####################################################################################################################################################
        
        # Area and Moment of Inertia (moi) attributes
        lamb_area= [25,100,500,1000,5000,10000,20000,50000,100000,150000]
        lamb_moi = [0.2, 0.3, 0.4, 0.5]

        ##########################################################################################
        # Extract training samples to compute the markers
        padding_size = 2
        window_size = padding_size+1+padding_size
        

        X_train = np.zeros((nb_samples_train,number_channels*window_size*window_size))
        y_train = np.zeros((nb_samples_train))
    
        k = 0    
        for i in train_indices:
          d1_train = zurich_training_images[i].shape[0]
          d2_train = zurich_training_images[i].shape[1] 
        
          zurich = zurich_training_images[i]
          zurich_padding = np.pad(zurich, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
          random_zurich_training_i = random_zurich_training_samples[i]
        
          for line in range(padding_size,d1_train+padding_size):
              for column in range(padding_size,d2_train+padding_size):
                  if (random_zurich_training_i[line-padding_size,column-padding_size] != 0):
                      X_train[k, :] = np.reshape(zurich_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                      y_train[k] = random_zurich_training_i[line-padding_size,column-padding_size]
                      k = k + 1

        
        unique,counts = np.unique(y_train,return_counts=True)
        print("Training samples: ", counts, unique)

        ################################################################################################################
        
        # Create the marker based on the results of the RF applied to the raw data (for the all training images and test images)     
        if (bool_prior_knowledge):
                 
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            
            # Apply to training images
            for i in zurich_training_images.keys():
                zurich = zurich_training_images[i]
                d1 = zurich.shape[0]
                d2 = zurich.shape[1]
                X_i = np.zeros((d1*d2,number_channels*window_size*window_size))
                zurich_padding = np.pad(zurich, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect') 
                k=0
                for line in range(padding_size,d1+padding_size):
                    for column in range(padding_size,d2+padding_size):
                        X_i[k, :] = np.reshape(zurich_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                        k = k + 1
                
                markers_ = clf.predict_proba(X_i)               
                zurich_training_markers[i] =  markers_

            # Apply to test images
            for i in zurich_test_images.keys():
                zurich = zurich_test_images[i]
                d1 = zurich.shape[0]
                d2 = zurich.shape[1]
                X_i = np.zeros((d1*d2,number_channels*window_size*window_size))
                zurich_padding = np.pad(zurich, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect') 
                k=0
                for line in range(padding_size,d1+padding_size):
                    for column in range(padding_size,d2+padding_size):
                        X_i[k, :] = np.reshape(zurich_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                        k = k + 1
                        
                markers_ = clf.predict_proba(X_i)
                zurich_test_markers[i] = markers_
                
        ################################################################################################################
        ################################################################################################################
        
        start_time=time.time()
        all_features = {}
        
        if   (method == "NIR+RGB"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = np.array([zurich[:,:,0],zurich[:,:,1],zurich[:,:,2],zurich[:,:,3]])
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = np.array([zurich[:,:,0],zurich[:,:,1],zurich[:,:,2],zurich[:,:,3]])     
                 
        elif   (method == "NIR+RGB_5x5"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                d1_train = zurich.shape[0]
                d2_train = zurich.shape[1] 
                padding_size = 2
                window = padding_size*2+1
                                
                zurich_padding = np.pad(zurich, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
              
                X_train = np.zeros((window*window*number_channels, d1_train, d2_train))
                for line in range(padding_size,d1_train+padding_size):
                    for column in range(padding_size,d2_train+padding_size):
                        X_train[:, line-padding_size, column-padding_size] = np.reshape(zurich_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                all_features[i] = X_train
                
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                d1_train = zurich.shape[0]
                d2_train = zurich.shape[1] 
                padding_size = 2
                window = padding_size*2+1
                                
                zurich_padding = np.pad(zurich, ((padding_size, padding_size), (padding_size, padding_size), (0,0)), 'reflect')
              
                X_test = np.zeros((window*window*number_channels, d1_train, d2_train))
                for line in range(padding_size,d1_train+padding_size):
                    for column in range(padding_size,d2_train+padding_size):
                        X_test[:, line-padding_size, column-padding_size] = np.reshape(zurich_padding[line-padding_size:line+padding_size+1,column-padding_size:column+padding_size+1,:], (number_channels*window_size*window_size))
                all_features[i] = X_test                          
                
                      
        elif (method == "AP"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_AP(zurich, lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_AP(zurich, lamb_area, lamb_moi, adj)   
        elif (method == "MAX"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_MAX(zurich, lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_MAX(zurich, lamb_area, lamb_moi, adj)   
        elif (method == "MIN"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_MIN(zurich, lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_MIN(zurich, lamb_area, lamb_moi, adj)   
        elif (method == "SDAP"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_SDAP(zurich, lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_SDAP(zurich, lamb_area, lamb_moi, adj)   
        elif (method == "ALPHA"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_ALPHA(zurich, zurich_training_markers[i], lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_ALPHA(zurich, zurich_test_markers[i], lamb_area, lamb_moi, adj)  
        elif (method == "OMEGA"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_OMEGA(zurich, zurich_training_markers[i], lamb_area, lamb_moi, adj)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_OMEGA(zurich, zurich_test_markers[i], lamb_area, lamb_moi, adj)  
        elif (method == "WATERSHED" or method == "WATERSHED_COMPUTED_WITH_PK"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_WATERSHED(zurich, zurich_training_markers[i], lamb_area, lamb_moi, adj, watershed_attribute, bool_prior_knowledge)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_WATERSHED(zurich, zurich_test_markers[i], lamb_area, lamb_moi, adj, watershed_attribute, bool_prior_knowledge)  
        elif (method == "WATERSHED_FILTERED_WITH_PK"):
            for i in zurich_training_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_training_images[i]
                all_features[i] = compute_WATERSHED_FILTERED_WITH_PK(zurich, zurich_training_markers[i],adj, watershed_attribute, 'same', 7, number_classes)
            for i in zurich_test_images.keys():
                print("Computing features of image ", i, "...")
                zurich = zurich_test_images[i]
                all_features[i] = compute_WATERSHED_FILTERED_WITH_PK(zurich, zurich_test_markers[i], adj, watershed_attribute, 'same', 7, number_classes)  

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
        for i in zurich_training_images.keys():
            training_gt = random_zurich_training_samples[i]
            d1,d2 = training_gt.shape[0],training_gt.shape[1]
            pos_samples = np.where(training_gt > 0)
            for j in range(len(pos_samples[0])):
                X_train[k, :] = all_features[i][:, pos_samples[0][j], pos_samples[1][j]]
                y_train[k] = random_zurich_training_samples[i][pos_samples[0][j], pos_samples[1][j]]
                k = k+1
         
        unique, counts = np.unique(y_train, return_counts=True)
        print("Training samples : ", unique,counts)
              
        print(np.sum(X_train < 0))  
        print("Start training...")  
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        print("Finish training...")
        
        
        ##############################################################
    
        all_y_test = []
        all_label_classif = {}
        
        # Compute evaluation results per test image
        for i in zurich_test_images.keys():
            test_gt = zurich_test_gt[i]
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
            if (i != list(zurich_test_images.keys())[0]):
                all_label_classif[iteration] = np.concatenate((all_label_classif[iteration], label_classif), axis=0) 
            else:
                all_label_classif[iteration] = label_classif

            # Plot the classification results        
            if (bool_only_buildings):
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
                plt.imsave(OUT_DIR+'result_buildings_'+method+"_"+str(adj)+"_"+watershed_attribute+"_"+str(window_size)+"_iteration_"+str(i)+"_"+str(iteration+1)+".jpg",classification_image,cmap=cmap_)     
            
            else:
                img_result_classification = np.zeros((d1,d2))
                k = 0
                for line in range(0,d1):
                    for column in range(0,d2):
                        if (test_gt[line,column] > 0):
                            img_result_classification[line, column] =  label_classif[k]
                            k=k+1            
                
                # white, black, gray, light green, dark green, brown, dark blue, yellow, violet
                array_colors = [(1,1,1,1),(0,0,0,1),(0.4,0.4,0.4,1),(0,0.5,0,1),(0,1,0,1),(0.6,0.3,0,1),(0,0,0.6,1),(1,1,0,1),(0.6,0.6,1,1)]
                unique = np.unique(img_result_classification)
                array_colors_aux = [array_colors[int(j)] for j in unique]
                cmap_ = colors.ListedColormap(array_colors_aux)
                plt.imshow(np.reshape(rankdata(img_result_classification, method='dense'), (d1,d2))-1, cmap_)
                plt.axis('off')
                plt.savefig(OUT_DIR+'result_'+method+'_'+str(adj)+'_'+watershed_attribute+'_'+str(i)+'_'+str(iteration+1)+'.png', bbox_inches='tight', pad_inches=0, dpi=150)
         

            OCA[i,iteration],kappa[i,iteration],perclass[i,iteration],ACA[i,iteration],ConfMat = classificationEvaluation(y_test,label_classif, number_classes)
          
        ##############################################################
        final_OCA[iteration],final_kappa[iteration],final_perclass[iteration],final_ACA[iteration], final_ConfMat = classificationEvaluation(all_y_test,all_label_classif[iteration], number_classes)
        
        # The mean Intersection Over Union is equal as : TP/ (TP + FP + FN)
        if (bool_only_buildings):
            final_meanIOU[iteration] = np.sum(np.minimum(all_y_test-1,all_label_classif[iteration]-1)>0) / np.sum(np.maximum(all_y_test-1,all_label_classif[iteration]-1)>0)               
        
        ##############################################################
        # Precision and recall
        
        for i in range(0,number_classes):
            # The Recall is defined as : TP / (TP + FN)
            per_class_recall[iteration,i] = final_ConfMat[i,i]/np.sum(final_ConfMat[i,:])
            # The Precision is defined as : TP / (TP + FP)
            per_class_precision[iteration,i] = final_ConfMat[i,i]/np.sum(final_ConfMat[:,i])
            if (per_class_recall[iteration,i] > 0 or per_class_precision[iteration,i] > 0):
                per_class_f1[iteration,i] = 2*per_class_precision[iteration,i]*per_class_recall[iteration,i]/(per_class_precision[iteration,i]+per_class_recall[iteration,i])

                         
    ##############################################################    
               
    print('\n\n')
    print("Results per image ")   
    for i in zurich_test_images.keys():    
    
        if (bool_prior_knowledge):
          print("Image", str(i), " ", method, ' ', watershed_attribute, " with prior knowlegde from markers")
        else:
          print("Image", str(i), " ", method, ' ', watershed_attribute)
        print(" OA: ", np.round(np.mean(OCA[i,:]*100),2), "+-", np.round(np.std(OCA[i,:]*100),2), ", AA: ", np.round(np.mean(ACA[i,:]*100),2), "+-", np.round(np.std(ACA[i,:]*100),2), ", Kappa: ", np.round(np.mean(kappa[i,:]*100),2), "+-", np.round(np.std(kappa[i,:]*100),2))
        print("Accuracy per class: ")
        for n in range(number_classes):
            print(np.round(np.mean(perclass[i,:,n]*100),2) , "+-",  np.round(np.std(perclass[i,:,n]*100),2), ", " , end='')
        print('\n\n')
    
        
    print("Final results: ")
    print(" OA: ", np.round(np.mean(final_OCA*100),2), "+-", np.round(np.std(final_OCA*100),2), ", AA: ", np.round(np.mean(final_ACA*100),2), "+-", np.round(np.std(final_ACA*100),2), ", Kappa: ", np.round(np.mean(final_kappa*100),2), "+-", np.round(np.std(final_kappa*100),2))
    print('\n')
    
    
    print("Accuracy per class: ")
    for n in range(number_classes):
        print(np.round(np.mean(final_perclass[:,n]*100),2) , "+-",  np.round(np.std(final_perclass[:,n]*100),2), ", " , end='')
    print('\n')
    
    ##############################################################
    print("Precision per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(per_class_precision[:,n]*100),2) , "+- ",  np.round(np.std(per_class_precision[:,n]*100),2), ", ", end='')
    print('\n')

    print("Recall per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(per_class_recall[:,n]*100),2) , "+- ",  np.round(np.std(per_class_recall[:,n]*100),2), ", ", end='')
    print('\n')
    
    print("F1 score per class: ")
    
    for n in range(number_classes):
        print(np.round(np.mean(per_class_f1[:,n]*100),2) , " +- ",  np.round(np.std(per_class_f1[:,n]*100),2), ", ",  end='')
    print('\n')
    
    print("Mean F1 score:")
    print(np.round(np.mean(per_class_f1*100)) , " +- ",  np.round(np.std(per_class_f1*100)), end='')
    print('\n')
    
    if (bool_only_buildings):
        print("Mean IoU: ")
        print(" & ", np.round(np.mean(final_meanIOU*100),2), "$\\pm$", np.round(np.std(final_meanIOU*100),2))

    ##############################################################
    
if __name__ == "__main__":
    main()