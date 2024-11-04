from deepxrte.geometry import Rectangle
import torch
from utils import read_csv, write_csv, charge_data, init_model
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()


############# VARIABLES ################

folder_result_name = "8_stronger"  # name of the result folder
folder_result = "results/" + folder_result_name


# test seed, keep the same to compare the results
random_seed_test = 2002


##### Hyperparameters
# Uniquement si nouveau modèle

hyper_param_init = {
    "nb_epoch": 2000,  # epoch number
    "save_rate": 50,  # rate to save
    "weight_data": 1,
    "weight_pde": 1,
    "batch_size": 5000,  # for the pde
    "nb_points_pde": 1000000,  # Total number of pde points
    "Re": 100,
    "lr_init": 1e-3,  # Learning rate at the begining of training
    "gamma_scheduler": 0.999,  # Gamma scheduler for lr
    "nb_layers": 10,
    "nb_neurons": 64,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "nb_points_axes": 15,  # le nombre de points pris par axe par pas de temps
    "x_min": 0.15,
    "x_max": 0.325,
    "y_min": -0.1,
    "y_max": 0.1,
    "t_min": 4,
    "t_max": 5,
    "transfert_learning": "None"
}


# Charging the model

Path(folder_result).mkdir(parents=True, exist_ok=True)  # Creation du dossier de result
if not Path(folder_result + "/hyper_param.json").exists():
    with open(folder_result + "/hyper_param.json", "w") as file:
        json.dump(hyper_param_init, file, indent=4)
    hyper_param = hyper_param_init

else:
    with open(folder_result + "/hyper_param.json", "r") as file:
        hyper_param = json.load(file)

##### The code ###############################
###############################################

# Data loading
X_train_np, U_train_np, X_full, U_full, mean_std = charge_data(hyper_param)
X_train = torch.from_numpy(X_train_np).requires_grad_().to(torch.float32).to(device)
U_train = torch.from_numpy(U_train_np).requires_grad_().to(torch.float32).to(device)


# le domaine de résolution
rectangle = Rectangle(
    x_max=X_full[:, 0].max(),
    y_max=X_full[:, 1].max(),
    t_min=X_full[:, 2].min(),
    t_max=X_full[:, 2].max(),
    x_min=X_full[:, 0].min(),
    y_min=X_full[:, 1].min(),
)


X_pde = rectangle.generate_lhs(hyper_param["nb_points_pde"]).to(device)

# Data test loading
torch.manual_seed(random_seed_test)
np.random.seed(random_seed_test)
X_test_pde = rectangle.generate_lhs(hyper_param["n_pde_test"]).to(device)
points_coloc_test = np.random.choice(
    len(X_full), hyper_param["n_data_test"], replace=False
)
X_test_data = torch.from_numpy(X_full[points_coloc_test]).to(device)
U_test_data = torch.from_numpy(U_full[points_coloc_test]).to(device)


# Initialiser le modèle


# On plot les print dans un fichier texte
with open(folder_result + "/print.txt", "a") as f:
    model, optimizer, scheduler, loss, train_loss, test_loss = init_model(
        f, hyper_param, device, folder_result
    )
    ######## On entraine le modèle
    ###############################################
    train(
        nb_epoch=hyper_param["nb_epoch"],
        train_loss=train_loss,
        test_loss=test_loss,
        poids=[hyper_param["weight_data"], hyper_param["weight_pde"]],
        model=model,
        loss=loss,
        optimizer=optimizer,
        X_train=X_train,
        U_train=U_train,
        X_pde=X_pde,
        X_test_pde=X_test_pde,
        X_test_data=X_test_data,
        U_test_data=U_test_data,
        Re=hyper_param["Re"],
        time_start=time_start,
        f=f,
        u_mean=mean_std["u_mean"],
        v_mean=mean_std["v_mean"],
        x_std=mean_std["x_std"],
        y_std=mean_std["y_std"],
        t_std=mean_std["t_std"],
        u_std=mean_std["u_std"],
        v_std=mean_std["v_std"],
        p_std=mean_std["p_std"],
        folder_result=folder_result,
        save_rate=hyper_param["save_rate"],
        batch_size=hyper_param["batch_size"],
        scheduler=scheduler,
    )

####### On save le model et les losses

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    },
    folder_result + "/model_weights.pth",
)
write_csv(train_loss, folder_result, file_name="/train_loss.csv")
write_csv(test_loss, folder_result, file_name="/test_loss.csv")
