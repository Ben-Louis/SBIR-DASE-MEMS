# pylint: disable=import-error,no-name-in-module
import torch
import os
import argparse
import json
import shutil

from models import networks
from utils import makedir

def str2bool(s):
    return s.lower() == "true"


# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--dataset", type=str, default="TUBerlin", choices=["TUBerlin", "sketchy"])
    parser.add_argument("--data_path", type=str, default="datasets/data", help="Path of data root")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=4)

    # model
    parser.add_argument('--network', type=str, default='resnext101')
    parser.add_argument('--f_dim', type=int, default=512)

    # loss
    parser.add_argument('--loss_type', type=str, default='mems')
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--test_distance', type=str, default="euclidean", choices=["euclidean", "angular"])

    # process control
    parser.add_argument("--phase", type=str, default="train", choices=("train", "test"))
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--annealing_step', type=int, default=100000)
    parser.add_argument('--pretrained_model', type=int, default=-1)

    # log
    parser.add_argument("--save_path", type=str, default="expr", help="Path of saved results during train/test")
    parser.add_argument('--model_save_step', type=int, default=20000)

    config = parser.parse_args()

    ############################################
    #               Postprocesses              #
    ############################################
    config.save_path = os.path.join("logs", *config.save_path.split(os.sep))
    config.log_path = os.path.join(config.save_path, "log")
    config.model_path = os.path.join(config.save_path, "model")
    config.vec_path = os.path.join(config.save_path, "vec")

    makedir(config.log_path)
    makedir(config.model_path)
    makedir(config.vec_path)

    return config


def main():
    config = get_parameter()

    # set gpu
    torch.backends.cudnn.benchmark = True
    config.device = torch.device('cuda:0')

    # build model
    model = {"net": networks[config.network](f_dim=config.f_dim)}

    # train
    if config.phase == 'train':
        train(config, model)
        config.pretrained_model = config.num_steps
        config.phase = 'test'

    # test
    test(config, model)


def train(config, model):
    from trainer import Trainer
    trainer = Trainer(config, model)
    trainer.train()


def test(config, model):
    from tester import Tester, HashTester
    tester = Tester(config, model)
    tester.run()
    hash_tester = HashTester(config, model)
    hash_tester.run()


if __name__ == "__main__":
    main()
