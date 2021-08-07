import torch
from tqdm import tqdm
from datasets import get_dataset
from models import losses
from utils import load_checkpoint, save_checkpoint, get_data_from_iter


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.build_dataset()
        self.build_loss()
        self.build_opts()

    def build_dataset(self):
        dataset = {
            'pho': get_dataset(domain='photo', phase='train', data_path=self.config.data_path,
                               dataset=self.config.dataset, image_size=self.config.image_size),
            'skt': get_dataset(domain='sketch', phase='train', data_path=self.config.data_path,
                               dataset=self.config.dataset, image_size=self.config.image_size)
        }
        self.data_loader = {key: value.get_loader(batch_size=self.config.batch_size,
                                                  num_workers=self.config.num_workers, shuffle=True,
                                                  drop_last=True) for key, value in dataset.items()}

        self.config.c_dim = len(dataset["skt"].cate_num)

    def build_loss(self):
        self.model[self.config.loss_type] = losses[self.config.loss_type](
            dim_feature=self.config.f_dim,
            num_classes=self.config.c_dim,
            **vars(self.config)
        )

    def build_opts(self):
        # network params
        param_groups = [{"params": self.model["net"].parameters(), "weight_decay": 0.0001,
                         "initial_lr": 0.0001}]
        # loss params
        loss_param = []
        for key, model in self.model.items():
            if key != "net": loss_param.extend(list(model.parameters()))
        param_groups.append({"params": loss_param, "initial_lr": 0.0001})

        # optimizor
        self.opt = torch.optim.Adam(param_groups, lr=0.0001)

        # learning rate schedule
        def schedule_func(step):
            alpha = float(step) / self.config.num_steps
            return (1 - alpha) * 2 if alpha > 0.5 else 1
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.opt, lr_lambda=schedule_func)

    def train(self):

        # load pretrained model
        if self.config.pretrained_model > 0:
            load_checkpoint(self.model, self.config.model_path, step=self.config.pretrained_model)

        # model to device
        for key in self.model.keys():
            self.model[key] = self.model[key].to(self.config.device)

        # data_iters
        iters = {
            'pho': iter(self.data_loader['pho']),
            'skt': iter(self.data_loader['skt'])
        }
        img, label = {}, {}

        ############################################
        #              Start Training              #
        ############################################
        index = tqdm(range(self.config.num_steps),
                     initial=max(self.config.pretrained_model, 0),
                     desc="start training")
        for i in index:
            # get data
            for key in ('pho', 'skt'):
                (img[key], label[key]), iters[key] = get_data_from_iter(iters[key], self.data_loader[key])

            # process feature and label
            imgs = torch.cat((img['skt'], img['pho'])).to(self.config.device)
            domains = torch.cat((torch.zeros(img['skt'].size(0), device=imgs.device),
                                 torch.ones(img['pho'].size(0), device=imgs.device)))
            feats = self.model['net'](imgs, domains)
            labels = torch.cat((label['skt'], label['pho'])).to(self.config.device)

            img.clear()
            label.clear()

            # compute loss
            loss = self.model[self.config.loss_type](feats, labels).mean()
            index.set_description("{0}: {1:.3f}".format(self.config.loss_type, loss.item()))

            # optimize parameters
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.lr_scheduler.step()

            # save model
            if (i+1) % self.config.model_save_step == 0:
                save_checkpoint(self.model, save_path=self.config.model_path, step=(i+1))