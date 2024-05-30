import pickle

with open('Additional Data/hanhchinh_50k_final.pkl', 'rb') as file:
    data_doisong = pickle.load(file)

print(len( data_doisong))