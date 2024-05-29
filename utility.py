import pickle
from tqdm import tqdm
import csv
import numpy as np

def save_testset(X_test_filename = 'testset_200\\test_X_200.pkl', Y_test_filename = 'testset_200\\test_Y_200.pkl', size = 200, max_length = 7, min_length = 4):
    X_test = pickle.load(open('data_small\\test_X_50k.pkl', 'rb'))
    Y_test = pickle.load(open('data_small\\test_Y_50k.pkl', 'rb'))
    test_X_new = []
    test_Y_new = []
    count = 0
    for i in tqdm(range(50000)):
        if count == size:
            break
        X_sequence = X_test[i]
        Y_sequence = Y_test[i]
        seq_len = len(X_sequence.split()) 
        if seq_len <= max_length and seq_len >= min_length:
            print(Y_sequence) # Debugging line
            test_X_new.append(X_sequence)
            test_Y_new.append(Y_sequence)
            count += 1
    with open(X_test_filename, 'wb+') as f:
        pickle.dump(test_X_new, f)
    with open(Y_test_filename, 'wb+') as f:
        pickle.dump(test_Y_new, f)

def save_testset_CSV(X_test_filename = 'testset_200\\test_X_200_1.csv', Y_test_filename = 'testset_200\\test_Y_200_1.csv', size = 200, max_length = 7, min_length = 4):
    X_test = pickle.load(open('data_small\\test_X_50k.pkl', 'rb'))
    Y_test = pickle.load(open('data_small\\test_Y_50k.pkl', 'rb'))
    test_X_new = []
    test_Y_new = []
    count = 0
    for i in tqdm(range(40000, 50000)):
        if count == size:
            break
        X_sequence = X_test[i]
        Y_sequence = Y_test[i]
        seq_len = len(X_sequence.split()) 
        if seq_len <= max_length and seq_len >= min_length:
            print(Y_sequence) # Debugging line
            test_X_new.append(X_sequence)
            test_Y_new.append(Y_sequence)
            count += 1

    # Open the CSV file in append mode
    with open(X_test_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the new single line to the CSV file
        for sequence in test_X_new:
            writer.writerow([sequence])

    # Open the CSV file in append mode
    with open(Y_test_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the new single line to the CSV file
        for sequence in test_Y_new:
            writer.writerow([sequence])

def import_CSV_to_pkl(X_test_csv = 'testset_200\\test_X_200_1.csv', Y_test_csv = 'testset_200\\test_Y_200_1.csv', X_test_pkl = 'testset_200\\test_X_200.pkl', Y_test_pkl = 'testset_200\\test_Y_200.pkl'):
    X_test = []
    Y_test = []
    with open(X_test_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in rows:
            X_test.append(row[0])
    with open(Y_test_csv, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in rows:
            Y_test.append(row[0])
            print(row[0])
    with open(X_test_pkl, 'wb+') as f:
        pickle.dump(X_test, f)
    with open(Y_test_pkl, 'wb+') as f:
        pickle.dump(Y_test, f)

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
   