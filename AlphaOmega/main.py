import argparse
import json
import os

import torch

from alphaomega.config import load_config
from alphaomega.lifecycle import Lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml"))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--layer-replacement", nargs=2, metavar=("THETA_0", "THETA_T"), default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.scenario, args.overrides)
    lifecycle = Lifecycle(cfg)
    if args.layer_replacement is None:
        lifecycle.run()
    else:
        Theta_0, Theta_t = (torch.load(path, map_location="cpu") for path in args.layer_replacement)
        print(json.dumps(lifecycle.layer_replacement(Theta_0, Theta_t), indent=2))


if __name__ == "__main__":
    main()
