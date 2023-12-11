import pickle

if __name__ == '__main__':

    data_list=[]
    labels_list=[]
    for individual in range(1,9):
        with open(f"data/HRD_all_class/individual{individual}/data.pkl", 'rb') as file:
            data_list.append(pickle.load(file))
        with open(f"data/HRD_all_class/individual{individual}/labels.pkl", 'rb') as file:
            labels_list.append(pickle.load(file))
    print(data_list)
