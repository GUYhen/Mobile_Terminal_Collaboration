import torch
import torch.nn.functional as F


@torch.no_grad()
def predictions(model, Theta, loader, device):
    model.load_state_dict(Theta)
    model.eval()
    correct, P_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        P = F.softmax(model(x), dim=1)
        correct.append((P.argmax(dim=1) == y).float().cpu())
        P_true.append(P.gather(1, y.unsqueeze(1)).squeeze(1).cpu())
    return torch.cat(correct), torch.cat(P_true)


def remaining_accuracy(model, Theta_prime, loader_r, device):
    correct, _ = predictions(model, Theta_prime, loader_r, device)
    return float(correct.mean())


def forgetting_accuracy(model, Theta_prime, loader_u, device):
    correct, _ = predictions(model, Theta_prime, loader_u, device)
    return float(correct.mean())


def forgetting_rate(model, Theta, Theta_prime, loader_u, device):
    _, P_Theta = predictions(model, Theta, loader_u, device)
    _, P_Theta_prime = predictions(model, Theta_prime, loader_u, device)
    return float(1.0 - (P_Theta_prime / P_Theta).mean())
