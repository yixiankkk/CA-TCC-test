import os

import numpy as np
import torch
from torch import nn

from dataloader.dataloader import data_generator, Load_Dataset
from models.TC import TC
from models.model import base_Model


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs


data_type = 'HRD'
training_mode = 'ft_SupCon_1per'
device = torch.device('cuda:0')
logs_save_dir='experiments_logs'
experiment_description='HRD_experiments'
run_description='test1'


SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# Load datasets
data_path = os.path.join(r'data/', data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)

batch_size = configs.batch_size
test_dl_list = []
for i in range(8):
    test_dataset = torch.load(os.path.join(data_path, 'test', "test_dict"+str(i)+".pt"))
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    test_dl_list.append(test_loader)

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

load_from = os.path.join(
    os.path.join(logs_save_dir, experiment_description, run_description, f"ft_SupCon_1per_seed_{SEED}", "saved_models"))
chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
pretrained_dict = chkpoint["model_state_dict"]
model.load_state_dict(pretrained_dict)

# Load temporal_contr_model
load_from = os.path.join(
    os.path.join(logs_save_dir, experiment_description, run_description, f"ft_SupCon_1per_seed_{SEED}", "saved_models"))
chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
pretrained_dict = chkpoint["temporal_contr_model_state_dict"]
temporal_contr_model.load_state_dict(pretrained_dict)

test_acc_list=[]
test_loss_list=[]

for dl in test_dl_list:
    test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, dl, device, training_mode)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

print(test_loss_list)
