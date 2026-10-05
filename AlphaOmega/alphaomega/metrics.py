import torch

from .utils import loader, pixel_ce


class ConfusionMatrix:
    def __init__(self, C, ignore_index):
        self.C = C
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(C, C, dtype=torch.long)

    def update(self, pred, y):
        valid = (y != self.ignore_index) & (y < self.C)
        index = (y[valid] * self.C + pred[valid]).cpu()
        self.matrix += torch.bincount(index, minlength=self.C ** 2).view(self.C, self.C)

    def IoU(self):
        m = self.matrix.double()
        intersection = m.diag()
        union = m.sum(0) + m.sum(1) - intersection
        return torch.where(union > 0, intersection / union.clamp_min(1), torch.full_like(intersection, float("nan")))


@torch.no_grad()
def evaluate(model, dataset, classes, batch_size, ignore_index, device):
    model.eval()
    cm = ConfusionMatrix(model.num_classes, ignore_index)
    loss, n = 0.0, 0
    for x, y, _ in loader(dataset, batch_size, False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += float(pixel_ce(logits, y, ignore_index)) * x.shape[0]
        n += x.shape[0]
        cm.update(logits.argmax(1), y)
    IoU = cm.IoU()[list(classes)]
    observed = IoU[~torch.isnan(IoU)]
    mIoU = float(observed.mean()) if observed.numel() else float("nan")
    return {"mIoU": mIoU, "IoU": IoU.tolist(), "loss": loss / max(n, 1)}
