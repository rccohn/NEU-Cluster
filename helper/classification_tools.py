import numpy as np


def label_matcher(y_cluster, labels, return_mapper=False):
    """
    maps cluster centers to true labels based on the most common filename for each cluster. 

    Parameters
    ----------
    y_cluster: ndarray
        n-element array of labels obtained from clusters
        
    labels: ndarray
        n-element array of ground truth labels for which y_cluster will be mapped to
        
    return_mapper:bool
        if True, dictionary mapping values in y_cluster to values in labels will be returned


    Returns
    -----------
    y_pred: ndarray
        n-element array of values in y_cluster mapped to labels
    
    mapper (optional): dict
        dictonary whose keys are elements of y_cluster and values are the corresponding
        elements of labels.

    """
    
    assert type(y_cluster) == np.ndarray and type(labels) == np.ndarray

    y_cluster_unique = np.unique(y_cluster)

    
    mapper = {}  # keys will be cluster ID's, values will be corresponding label
    
    for x in y_cluster_unique:
        unique, counts = np.unique(labels[y_cluster==x], return_counts=True)  # get frequency of each gt label in cluster x
        mapper[x] = unique[counts.argmax()]  # set mapper[x] to the most frequent label in the cluster

    y_pred = np.asarray([mapper[x] for x in y_cluster])  # map cluster id's to labels

    if return_mapper:
        return y_pred, mapper
    else:
        return y_pred

