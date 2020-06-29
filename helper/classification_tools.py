import numpy as np

from sklearn.metrics import classification_report

#TODO module docstring, examples for functions

class CustomLabelEncoder:
    """
    Creates a mapping between string labels and integer class clabels for working with categorical data.
    
    
    Attributes
    ----------
    mapper:None dict
        None if mapper is not supplied or model is not fit.
        keys are unique string labels, values are integer class labels.
    """
    def __init__(self, mapper=None):
        """
        Initializes class instance.
        
        If the mapper dictionary is supplied here, then the model can be used without calling .fit().
        
        Parameters
        -----------
        mapper (optional): dict or None
            if mapper is None encoder will need to be fit to data before it can be used.
            If it is a dictionary mapping string labels to integer class labels, then this will be stored
            and the model can be used to transform data.
        """
        self.mapper = mapper
    
    def fit(self, str_labels, sorter=None):
        """
        Fits string labels to intiger indices with optional sorting.
        
        np.unique() is used to extract the unique values form labels. If 
        
        Parameters
        ----------
        str_labels: list-like
            list or array containing string labels
        
        sorter (optional): None or function
            key for calling sorted() on data to determine ordering of the numeric indices for each label.
            
        Attributes
        -----------
        mapper: dict
            dictionary mapping string labels to the sorted integer indices is stored after fitting.
        
        """
        sorted_unique = sorted(np.unique(str_labels), key=sorter)
        mapper = {label: i for i, label in enumerate(sorted_unique)}
        self.mapper = mapper    

    def transform(self, str_labels):
        """
        Maps string labels to integer labels.
        
        Parameters
        ----------
        str_labels: list-like
            list of string labels whose elements are in self.mapper
        
        Returns
        --------
        int_labels: array
            array of integer labels  corresponding to the string labels
        """
        assert self.mapper is not None, 'Encoder not fit yet!'
        
        int_labels = np.asarray([self.mapper[x] for x in str_labels], np.int)
        
        return int_labels
        
    def inverse_transform(self, int_labels):
        """
        Maps integer labels to original string labels.
        
        Parameters
        -----------
        int_labels: list-like
            list or array of integer class indices
        
        Returns
        ----------
        str_labels: array(str)
            array of string labels corresponding to intiger indices
        
        """
        assert self.mapper is not None, 'Encoder not fit yet!'
        
        reverse_mapper = {y:x for x,y in self.mapper.items()}
        
        str_labels = np.asarray([reverse_mapper[x] for x in int_labels])
        
        return str_labels
    
    @property
    def labels_ordered(self):
        """
        Returns an array containing the string labels in order of which they are stored.
        
        For example, if the label_encoder has the following encoding: {'a':1,'c':3,'b':2},
        then this will return array(['a','b','c'])
        """
        pass
    
    @labels_ordered.getter
    def labels_ordered(self):
        return self.inverse_transform(range(len(self.mapper)))


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
        
    y_cluster = np.asarray(y_cluster)
    labels = np.asarray(labels)
    
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


def label_matcher_multicluster(y_cluster, labels_numeric, le, return_mapper=False):
    """
    label_matcher_multicluster(y_cluster, labels_numeric, le, [return_mapper]):
    
    Maps cluster labels to ground truth labels, keeping track of the case where multiple clusters map to one gt class.
    
    Matches cluster labels to gt class labels using label_matcher. For each gt label, check if there are multiple clusters
    mapped to that label. If so, append integers to identify unique clusters.
    For example, if cluster labels 0,1,2 map to 'Cr','In',and 'In', maps them to 'Cr','In-0', and 'In-1'.
    
    Parameters
    -----------
    y_cluster: ndarray
        n-element array of predicted labels obtained from clustering
    
    labels_numeric: ndarray
        n-element array of numeric ground truth labels
    
    le: LabelEncoder object
        Used to transform labels_numeric to string labels.
    
    return_mapper: bool
        If True, a dictionary mapping cluster labels to the unique string label will be returned.
        Default: False
    
    Returns
    ---------
    labels_multicluster: list
        n-element list of string labels with integer identifiers distinguishing individual clusters
    
    mapper (optional): dict
        dictionary whose keys are unique values in y_cluster and values are the corresponding
        string label with unique cluster identifier. Ex {0: 'Cr', 1: 'In-1', 2: 'In-2'}
    
    """
    
    y_pred, label_mapper = label_matcher(y_cluster, labels_numeric, return_mapper=True)
    
    mapper = {}
    for gt_label in np.unique(list(label_mapper.values())):
        gt_str_label = le.inverse_transform([gt_label])[0]
        cluster_labels = sorted([k for k, v in label_mapper.items() if v == gt_label])

        if len(cluster_labels) == 1:
            mapper[cluster_labels[0]] = gt_str_label
        else:            
            for i, yp in enumerate(cluster_labels, 1):
                mapper[yp] = "{}-{}".format(gt_str_label, i)
    
    labels_multicluster = [mapper[x] for x in y_cluster]
    
    if return_mapper:
        return labels_multicluster, mapper
    else:
        return labels_multicluster
         
   
def latex_report(y_true, y_pred, target_names, include_head=True):
    """
    Computes precision and recall for each class and outputs results to a string to be compiled as a LaTeX table. 
    
    Parameters
    ----------
    y_true, y_pred: ndarray
        array of ground truth (y_true) and predicted (y_pred) integer labels
    
    target_names: list
        list of strings corresponding to each label in y_true and y_pred. The index corresponds to the integer label.
    
    include_head: bool
        If True, includes \begin{tabular} and \end{tabular} lines. Default: True.
    
    Returns
    --------
    table: str
        string output that can be compiled into a LaTeX table.
    
    """
    #TODO add begin{tabular} and end{tabular}
    data = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    table = ''
    if include_head:
        table = ''.join([table, '\\begin{tabular}{lcc}\n'])
    
    table = ''.join([table, '\tClass\t& Precision\t& Recall\t\\\\  \\hline \n'])
    for key in target_names:
        line = '\t{}\t& {:.3f} \t& {:.3f} \t\\\\\n'.format(key, np.round(data[key]['precision'], decimals=3), np.round(data[key]['recall'], decimals=3))
        table = ''.join([table, line])
    
    if include_head:
        table = ''.join([table, '\\end{tabular}'])
        
    return table    


def deconstruct_cm(cm):
    """
    Turns 2d confusion matrix into 2 1d arrays of ground truth values and predicted values.
    
    In some cases the cache saves confusion matrices instead of the actual predictions, but it is more
    convenient to have predictions for the classification report. Note that the order of
    files is not preserved.
    
    Parameters
    ----------
    cm: ndarray
        n_class x n_class array of ints corresponding to the number of predictions for each
        class.
    
    Returns
    --------
    gt, pred: ndarray
        n_class element array of integer ground truth (gt) or predicted (pred) labels
        
    
    
    Examples
    --------
    
    cm = np.asarray([[1,2],[3,4]])
    gt, pred = deconstruct_cm(cm)
    print(gt)
    >>> [0 0 0 1 1 1 1 1 1 1]
    print(pred)
    >>> [0 1 1 0 0 0 1 1 1 1]
    """
    r, c = cm.shape
    assert r == c
    gt = []
    pred = []
    for i in range(r):
        for j in range(c):
            for _ in range(cm[i,j]):
                gt.append(i)
                pred.append(j)
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    return gt, pred

