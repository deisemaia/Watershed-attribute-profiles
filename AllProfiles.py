import numpy as np
import higra as hg
import sap

################################################################################################################

def compute_AP(image, lamb_area, lamb_moi, adj):
    number_channels = image.shape[2]
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
    number_channels = image.shape[2]
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
    number_channels = image.shape[2]
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
    number_channels = image.shape[2]
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
    number_channels = image.shape[2]
    ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
   
    ALPHA_profile = sap.concatenate((ALPHA_area,ALPHA_moi))
    
    for i in range(1,number_channels):
        ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
        ALPHA_profile = sap.concatenate((ALPHA_profile,ALPHA_area,ALPHA_moi))
    
    final_ALPHA = sap.vectorize(ALPHA_profile)
    
    return final_ALPHA

def compute_OMEGA(image, markers, lamb_area, lamb_moi, adj):
    number_channels = image.shape[2]
    OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
    OMEGA_profile = sap.concatenate((OMEGA_area,OMEGA_moi))
    
    for i in range(1,number_channels):
        OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='max')
        OMEGA_profile = sap.concatenate((OMEGA_profile,OMEGA_area,OMEGA_moi))
    
    final_OMEGA = sap.vectorize(OMEGA_profile)
    
    return final_OMEGA

def compute_WATERSHED(image, markers, lamb_area, lamb_moi, adj, watershed_attribute, bool_prior_knowledge):
    number_channels = image.shape[2]
    if (bool_prior_knowledge):
        markers_ = np.sqrt(np.sum(markers*markers,axis=1))
        markers_ = 1 - ((markers_ - np.min(markers_))/(np.max(markers_) - np.min(markers_)))
        markers = np.reshape(markers_, (image.shape[0], image.shape[1]))
    else:
        markers = np.ones((image.shape[0], image.shape[1]))
    
    print("Shape of markers: ", markers.shape, np.unique(markers))
    WATERSHED_area = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, np.ascontiguousarray(markers), adjacency=adj, watershed_attribute=watershed_attribute)
    WATERSHED_moi = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, np.ascontiguousarray(markers), adjacency=adj,filtering_rule='max', watershed_attribute=watershed_attribute)
    WATERSHED_profile = sap.concatenate((WATERSHED_area,WATERSHED_moi))

    for i in range(1,number_channels):
        WATERSHED_area = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, np.ascontiguousarray(markers), adjacency=adj, watershed_attribute=watershed_attribute)
        WATERSHED_moi = sap.profiles.watershed_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, np.ascontiguousarray(markers), adjacency=adj,filtering_rule='max', watershed_attribute=watershed_attribute)          
        WATERSHED_profile = sap.concatenate((WATERSHED_profile,WATERSHED_area,WATERSHED_moi))
    
    final_WATERSHED = sap.vectorize(WATERSHED_profile)

    return final_WATERSHED
    
    
def compute_WATERSHED_FILTERED_WITH_PK(image, markers, adj, watershed_attribute, out_feature, number_thresh, number_classes):
    number_channels = image.shape[2]
    markers = np.reshape(markers, (image.shape[0], image.shape[1], number_classes))
    final_result = np.ones((number_channels*number_classes*number_thresh+number_channels, image.shape[0], image.shape[1]))
    pos = 0
    for i in range(0,number_channels):
        for j in range(0, number_classes):
            print(image[:,:,i].shape)
            image_ = np.copy(image[:,:,i])
            graph = hg.get_4_adjacency_graph(image_.shape) 
            weight = hg.weight_graph(graph, image_, hg.WeightFunction.L1)
            if (watershed_attribute=="area"):
                tree, alt = hg.watershed_hierarchy_by_area(graph, weight)
            elif (watershed_attribute=="dynamics"):
                tree, alt = hg.watershed_hierarchy_by_dynamics(graph, weight)
            elif (watershed_attribute=="volume"):
                tree, alt = hg.watershed_hierarchy_by_volume(graph, weight)
            
            marker  = markers[:,:, j]
            
            marker_min = np.amin(marker)
            marker_max = np.amax(marker)
            step = (marker_max - marker_min)/number_thresh
            #print("Min and max of marker: ", marker_min, marker_max)
            thresh = marker_min
            
            aux = 0
            while (aux < number_thresh):
                marker_thresh = np.reshape((marker > thresh)*1, (image[:,:,i].shape[0]*image[:,:,i].shape[1]))
                leaf_with_labels = hg.binary_labelisation_from_markers(tree, marker_thresh, (1-marker_thresh), graph)       
                
                attrib_max = hg.accumulate_sequential(tree, leaf_with_labels, hg.Accumulators.max)
                nodes_to_be_removed = (attrib_max == 0)*1
                attrib, variance = hg.attribute_gaussian_region_weights_model(tree, image_)
                reconstructed_image = hg.reconstruct_leaf_data(tree, attrib, deleted_nodes=nodes_to_be_removed, leaf_graph=graph)
                unique, counts = np.unique(reconstructed_image, return_counts=True)
                
                thresh = thresh+step
                
                final_result[pos,:,:] = reconstructed_image
                pos = pos+1
                aux = aux+1
    for i in range(0,image.shape[2]):   
        final_result[pos,:,:] = image[:,:,i]
        pos = pos+1
        
    return final_result

