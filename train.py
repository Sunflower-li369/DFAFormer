import argparse
import logging
import os
import random
import warnings
from pydoc import locate

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="./data/Synapse/train_npz",
    # default="/ljx/data3/DAEFormer/data/Synapse/train_npz",   # gai 路径
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="./data/Synapse/test_vol_h5",
    # default="/ljx/data3/DAEFormer/data/Synapse/test_vol_h5",   # gai 路径
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")  # gai 2.3 90000→190000
parser.add_argument("--max_epochs", type=int, default=200, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=20, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=20, help="eval_interval")    # 模型训练时每隔多少个epoch对验证集进行一次评估
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=2, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
# parser.add_argument(
#     "--module", default="networks.DAEFormer_DIFF_decoder_SGFN.DAEFormer.DAEFormer",help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
# )   # gai 12.23 module default
parser.add_argument(
    "--module", default="networks.DAEFormer_dual_decoder_SGFN.DAEFormer",help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)   # gai 12.23 module default

# gai 9.26: 添加 --use_patch 参数
# parser.add_argument("--use_patch", action="store_true", help="Use patch-based training with full-res input")
parser.add_argument('--use_patch', action='store_true',
                    help='use random patch cropping during training')

args = parser.parse_args()

# gai 10.8
if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    transformer = locate(args.module)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = transformer(num_classes=args.num_classes).cuda()
    # net = transformer(num_classes=args.num_classes, image_size=args.img_size).cuda()  # gai 9.26

    trainer = {
        "Synapse": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir)

