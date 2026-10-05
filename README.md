# Mobile Terminal Collaboration

This repository holds code for four methods on resource management for mobile terminal collaboration with incomplete information. The code comes from the Wireless Networking and Mobile Computing Laboratory at South Dakota State University (PI: Jun Huang).

| Folder | Method | Entry point |
|---|---|---|
| `RACE` | RACE, a federated learning platooning framework | `python main.py --config configs/default.yaml --method race` |
| `SCALE` | SCALE: Sensitivity-Aware Federated Unlearning with Information Freshness Optimization for Mobile Edge Computing | `python main.py --dataset fashionmnist --model lenet --tau client` |
| `AlphaOmega` | AlphaOmega, federated learning across rovers | `python main.py --config configs/default.yaml` |
| `RELIEF` | RELIEF | `python main.py --config configs/default.yaml` |

Run each command from inside its folder.

## Options

- **`RACE`**
  - `--method` takes `race` (the proposed method), `maddqn` or `convex_greedy`.
  - `--arch` takes `tsfen` or `mlp`.
  - `--N` and `--K` override the corresponding values in the config.
- **`SCALE`**
  - `--dataset` takes `fashionmnist` or `cifar100`.
  - `--model` takes `lenet`, `mobilenetv3` or `resnet18`.
  - `--tau` sets what is unlearned: `client`, `class` or `sample`.
  - `--n` overrides the corresponding config value.
- **`AlphaOmega`**
  - `--scenario` selects one of the scenarios defined in `configs/default.yaml`.
  - `--set KEY=VALUE ...` overrides config entries.
- **`RELIEF`**
  - `--method` selects the method.
  - `--set KEY=VALUE` overrides a config entry; it can be repeated.

## Setup

```
pip install -r requirements.txt
```

The code uses Python 3 with PyTorch. Install the PyTorch build that matches your CUDA version from https://pytorch.org.

## Datasets

The datasets are not included. Download them and place them where each config expects. Every path is relative to the project folder.

| Folder | Datasets | Default location |
|---|---|---|
| `RACE` | AI4Mars (MSL) | `./data/ai4mars-dataset-merged-0.1/msl` |
| `SCALE` | Fashion-MNIST, CIFAR-100 (torchvision format; automatic download is off) | `./data` |
| `AlphaOmega` | AI4MARS, MarsScapes, S5Mars | `data/AI4MARS`, `data/MarsScapes`, `data/S5Mars` |
| `RELIEF` | PAMAP2 | `./data/PAMAP2_Dataset/Protocol` |

Results are written to each project's `runs/` or `outputs/` folder.

## License

BSD 3-Clause; see [LICENSE](LICENSE).

## Acknowledgment

This material is based upon work supported by the National Science Foundation under Grant No. 2348422. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
