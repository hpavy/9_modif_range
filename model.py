from deepxrte.gradients import gradient, derivee_seconde
import torch
import torch.nn as nn


def pde(U, input, Re, x_std, y_std, u_mean, v_mean, p_std, t_std, u_std, v_std):
    # je sais qu'il fonctionne bien ! Il a été vérifié
    """Calcul la pde

    Args:
        U (_type_): u,v,p calcullés par le NN
        input (_type_): l'input (x,y,t)
    """
    u = U[:, 0].reshape(-1, 1)
    v = U[:, 1].reshape(-1, 1)
    p = U[:, 2].reshape(-1, 1)
    u_x = gradient(U, input, i=0, j=0, keep_gradient=True).reshape(-1, 1)
    u_y = gradient(U, input, i=0, j=1, keep_gradient=True).reshape(-1, 1)
    p_x = gradient(U, input, i=2, j=0, keep_gradient=True).reshape(-1, 1)
    p_y = gradient(U, input, i=2, j=1, keep_gradient=True).reshape(-1, 1)
    u_t = gradient(U, input, i=0, j=2, keep_gradient=True).reshape(-1, 1)
    v_x = gradient(U, input, i=1, j=0, keep_gradient=True).reshape(-1, 1)
    v_y = gradient(U, input, i=1, j=1, keep_gradient=True).reshape(-1, 1)
    v_t = gradient(U, input, i=1, j=2, keep_gradient=True).reshape(-1, 1)
    u_xx = derivee_seconde(u, input, j=0).reshape(-1, 1)
    u_yy = derivee_seconde(u, input, j=1).reshape(-1, 1)
    v_xx = derivee_seconde(v, input, j=0).reshape(-1, 1)
    v_yy = derivee_seconde(v, input, j=1).reshape(-1, 1)
    equ_1 = (
        (u_std / t_std) * u_t
        + (u * u_std + u_mean) * (u_std / x_std) * u_x
        + (v * v_std + v_mean) * (u_std / y_std) * u_y
        + (p_std / x_std) * p_x
        - (1 / Re) * ((u_std / (x_std**2)) * u_xx + (u_std / (y_std**2)) * u_yy)
    )
    equ_2 = (
        (v_std / t_std) * v_t
        + (u * u_std + u_mean) * (v_std / x_std) * v_x
        + (v * v_std + v_mean) * (v_std / y_std) * v_y
        + (p_std / y_std) * p_y
        - (1 / Re) * ((v_std / (x_std**2)) * v_xx + (v_std / (y_std**2)) * v_yy)
    )
    equ_3 = (u_std / x_std) * u_x + (v_std / y_std) * v_y
    return equ_1, equ_2, equ_3


## Le NN


class PINNs(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(3, hyper_param["nb_neurons"])])
        self.hiden_layers = nn.ModuleList(
            [
                nn.Linear(hyper_param["nb_neurons"], hyper_param["nb_neurons"])
                for _ in range(hyper_param["nb_layers"] - 1)
            ]
        )
        self.final_layer = nn.ModuleList([nn.Linear(hyper_param["nb_neurons"], 3)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    hyper_param = {"nb_layers": 0, "nb_neurons": 32}
    piche = PINNs(hyper_param)
    print(piche)
