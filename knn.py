# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:53:10 2025

@author: krsko
"""

import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        """Initialize k-NN classifier with given k and distance metric."""
        self.k = k
        self.metric = metric
    
    def fit(self, X_train, y_train):
        """Store the training data."""
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """Predict class labels for test data."""
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        """Compute distances and return the most common class label among k nearest neighbors."""
        if self.metric == 'euclidean':
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        else:
            raise ValueError("Unsupported distance metric")
        
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the class labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
