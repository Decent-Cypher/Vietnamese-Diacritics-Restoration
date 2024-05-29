import pickle
from tqdm import tqdm
import csv
import numpy as np



def accuracy_score(Y_true, Y_pred):
    """
    :param Y_true: A 2 dimensional numpy array of true labels.
    :param Y_pred: A 2 dimensional numpy array of predicted labels.
    Y_true and Y_pred must have the same shape.
    Labels can be strings or numbers.
    """
    total_words = np.count_nonzero(Y_true)
    assert Y_pred.shape == Y_true.shape 

    # Create a boolean mask for the non-zero elements in both arrays
    non_zero_mask1 = Y_true != 0
    non_zero_mask2 = Y_pred != 0

    # Combine the masks to find common non-zero elements
    combined_non_zero_mask = non_zero_mask1 & non_zero_mask2

    # Use the mask to compare elements and count equal non-zero elements
    correct_words = np.sum((Y_true == Y_pred) & combined_non_zero_mask)
    # print(total_words)
    # print(correct_words)
    return correct_words/total_words
   