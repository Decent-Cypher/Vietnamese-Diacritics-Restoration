import pickle
from tqdm import tqdm
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from InputHandler import InputHandler

def get_accuracy(list_inputs:list[str], list_outputs:list[str]):
    assert len(list_inputs) == len(list_outputs)
    IH_inp = InputHandler()
    list_inputs = IH_inp.remover(list_inputs)
    IH_out = InputHandler()
    list_outputs = IH_out.remover(list_outputs)
    count_true = 0
    count_total = 0
    for i in range(len(list_inputs)):
        assert len(list_inputs[i]) == len(list_outputs[i])
        for j in range(len(list_inputs[i])):
            count_total += 1
            count_true += int(list_inputs[i][j] == list_outputs[i][j])
    return count_true/count_total

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

# Y_pred_tokenized = [["tôi", "đang"], ["ăn"]]  
# # Y_pred_padded = pad_sequences(Y_pred_tokenized, padding='post', dtype=object)
# model_filename = "N_gram_model\\kneserney_trigram_administrative_cutoff1.pkl"
# model = pickle.load(open(model_filename, 'rb'))
# text = ["Luat Giao thong duoc doi moi", 'Lao Hac bi benh va qua doi']
# true = ["Luật Giao thông được đổi mới", 'Lão Hạc bị bệnh và qua đời']
# out = predict(model_filename, text)
# h = InputHandler()
# Y_pred_padded = pad_sequences(h.remover(out), padding='post', dtype=object)
# h1 = InputHandler()
# Y_true_padded = pad_sequences(h1.remover(true), padding='post', dtype=object)
# print(out)
# print(accuracy_score(Y_true_padded, Y_pred_padded))

   