from glob import glob
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.pt")))
    npzfiles = np.asarray(files , dtype='<U200')
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data




data_dir = r"data/HRD"
main_output_dir = "folds_data"
n_folds = 1
files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files if ".pt" in i])
files.sort()


folds = load_folds_data(data_dir, n_folds)
few_lbl_percentages = [1, 5, 10, 50, 75]

for fold_id in range(len(folds)):
    for percentage in few_lbl_percentages:
        output_dir = os.path.join(main_output_dir, f"fold_{fold_id}")
        os.makedirs(output_dir, exist_ok=True)

        ######## TRAINing files ##########
        training_files = folds[fold_id][1]

        # load files
        new_file_x = torch.load(training_files[0])["samples"]
        new_file_y = torch.load(training_files[0])["labels"]
        X_val, X_train, y_val, y_train  = train_test_split(new_file_x, new_file_y, test_size=percentage / 100, random_state=0)
     
        for np_file in training_files[1:]:
            new_file_x = torch.load(np_file)["samples"]
            new_file_y = torch.load(np_file)["labels"]
            X_val_n, X_train_n, y_val_n, y_train_n = train_test_split(new_file_x, new_file_y, test_size=percentage / 100,
                                                                  random_state=0)
            X_train = np.vstack((X_train, X_train_n))
            y_train = np.append(y_train, y_train_n)

        data_save = dict()
        X_train = np.transpose(X_train, [0,2,1])
        data_save["samples"] = torch.from_numpy(X_train)
        data_save["labels"] = torch.from_numpy(y_train)
        torch.save(data_save, os.path.join(output_dir, f"train_{fold_id}_{percentage}per.pt"))


    # load files
    X_train = torch.load(training_files[0])["samples"]
    y_train = torch.load(training_files[0])["labels"]

    for np_file in training_files[1:]:
        X_train = np.vstack((X_train, torch.load(np_file)["samples"]))
        y_train = np.append(y_train, torch.load(np_file)["labels"])

    data_save = dict()
    X_train = np.transpose(X_train, [0,2,1])
    data_save["samples"] = torch.from_numpy(X_train)
    data_save["labels"] = torch.from_numpy(y_train)
    torch.save(data_save, os.path.join(output_dir, f"train_{fold_id}_100per.pt"))

    ######## Validation ##########
    validation_files = folds[fold_id][1]
    # load files
    X_train = torch.load(validation_files[0])["samples"]
    y_train = torch.load(validation_files[0])["labels"]

    for np_file in validation_files[1:]:
        X_train = np.vstack((X_train, torch.load(np_file)["samples"]))
        y_train = np.append(y_train, torch.load(np_file)["labels"])

    data_save = dict()
    X_train = np.transpose(X_train, [0,2,1])
    data_save["samples"] = torch.from_numpy(X_train)
    data_save["labels"] = torch.from_numpy(y_train)
    torch.save(data_save, os.path.join(output_dir, f"val_{fold_id}.pt"))
