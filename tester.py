import os
import torch
import json
from tqdm import tqdm
from collections import defaultdict

from datasets import get_dataset
from utils import load_checkpoint, SpectralNorm, flip


class Tester:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.buffer = defaultdict(dict)

    def vectors(self, key):
        if key not in self.buffer["vectors"]:
            path = os.path.join(self.config.vec_path, "{0}_test_{1}.vec".format(key, self.config.pretrained_model))
            if os.path.exists(path):
                self.buffer["vectors"][key] = torch.load(path, map_location="cpu")
            else:
                # Prepare Model
                load_checkpoint(self.model, self.config.model_path, step=self.config.pretrained_model)
                self.model["net"].to(self.config.device)
                self.model["net"].eval()

                # prepare Data
                dataset = get_dataset(domain=key, phase="test", data_path=self.config.data_path,
                                      dataset=self.config.dataset, image_size=self.config.image_size)
                data_loader = dataset.get_loader(batch_size=self.config.batch_size, num_workers=self.config.num_workers)

                # Compute Feature
                feats, labels = [], []
                domain = torch.ones(1, device=self.config.device).float() * int(key == "photo")

                for x, y in tqdm(data_loader, desc=f"[test]compute features of {key}"):
                    x = x.to(self.config.device)
                    labels.append(y)
                    if x.dim() == 3: x = x.unsqueeze(0)
                    x = torch.cat((x, flip(x)))

                    with torch.no_grad():
                        feat = self.model["net"](x, domain.expand(x.size(0)))
                        feat = torch.stack(feat.split(feat.size(0) // 2, dim=0), dim=2)

                    feats.append(feat)

                # concate
                feats = torch.cat(feats).cpu()
                labels = torch.cat(labels)
                self.buffer["vectors"][key] = (feats, labels)

                # Save Features
                path = os.path.join(self.config.vec_path, '{0}_test_{1}.vec'.format(key, self.config.pretrained_model))
                torch.save(self.buffer["vectors"][key], path)
                
        return self.buffer["vectors"][key]

    def run(self):
        vectors = {"skts_feats": self.vectors("sketch")[0].to(self.config.device),
                   "skts_labels": self.vectors("sketch")[1],
                   "phos_feats": self.vectors("photo")[0].to(self.config.device),
                   "phos_labels": self.vectors("photo")[1]}
        mean_average_precision = self.compute_MAP(vectors)
        
        # save and display MAP
        info = "step: {0} | test MAP: {1:.3f}".format(self.config.pretrained_model, mean_average_precision)
        print(info)
        with open(os.path.join(self.config.log_path, 'result.txt'), 'a+') as f:
            f.write(info + '\n')

    def compute_distance(self, skt_feat, pho_feat):
        with torch.no_grad():
            if self.config.test_distance == "euclidean":
                dist1 = (skt_feat.narrow(2, 0, 1) - pho_feat).pow(2).sum(dim=1)
                dist2 = (skt_feat.narrow(2, 1, 1) - pho_feat).pow(2).sum(dim=1)
            elif self.config.test_distance == "angular":
                dist1 = torch.nn.functional.cosine_similarity(skt_feat.narrow(2, 0, 1), pho_feat, dim=1).acos()
                dist2 = torch.nn.functional.cosine_similarity(skt_feat.narrow(2, 1, 1), pho_feat, dim=1).acos()
            return torch.cat([dist1, dist2], dim=1).min(dim=1)[0].cpu()

    def compute_MAP(self, vectors):
        average_precisions = []
        for i in tqdm(range(len(vectors["skts_feats"])), desc="computing average precisions"):
            skt_feat = vectors["skts_feats"].narrow(0, i, 1)
            dist = self.compute_distance(skt_feat, vectors["phos_feats"])
            c = vectors["skts_labels"][i].item()
            _, index = dist.sort(dim=0)
            pos = (vectors["phos_labels"][index] == c).nonzero(as_tuple=False).squeeze()
            AP = (torch.arange(1, len(pos) + 1, dtype=torch.float32) / (pos.float() + 1)).mean().item()
            average_precisions.append((AP, c, i))
        return sum([t[0] for t in average_precisions]) / len(average_precisions) * 100


class HashTester(Tester):
    def run(self):
        self.centers = self.load_centers()
        for code_length in (32, 64, 128):
            self.dim_reduction_mapping = self.train_mapping(code_length)
            vectors = {"skts_feats": self.vectors("sketch")[0].to(self.config.device),
                       "skts_labels": self.vectors("sketch")[1],
                       "phos_feats": self.vectors("photo")[0].to(self.config.device),
                       "phos_labels": self.vectors("photo")[1]}

            mean_average_precision = self.compute_MAP(vectors)
            # save and display MAP
            info = "step: {0} | code length: {1} | hash MAP: {2:.3f}".format(self.config.pretrained_model,
                                                                             code_length, mean_average_precision)
            print(info)
            with open(os.path.join(self.config.log_path, 'result.txt'), 'a+') as f:
                f.write(info + '\n')
            self.buffer.clear()

    def load_centers(self):
        loss_saved_file_name = "model-{0}-{1}.cpkt".format(self.config.loss_type, self.config.pretrained_model)
        module = torch.load(os.path.join(self.config.model_path, loss_saved_file_name), map_location="cpu")
        return module["centers"]
        
    def train_mapping(self, code_length):
        centers = self.centers.to(self.config.device)
        if self.config.test_distance == "angular":
            centers = torch.nn.functional.normalize(centers, dim=1)
        c_dim, f_dim = self.centers.size()
        
        fc = SpectralNorm(torch.nn.Linear(f_dim, code_length))
        fc.to(self.config.device)
        opt = torch.optim.Adam(fc.parameters(), lr=1e-3)

        index = tqdm(range(10000))
        for step in index:
            code = fc(centers)
            code = code / (code.abs().detach() + 1e-8)
            loss = (code.mm(code.t())[(torch.ones(c_dim, c_dim)-torch.eye(c_dim)).to(self.config.device) > 0]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            index.set_description("training mapping, code length:{0} | loss: {1:.4f}".format(code_length, loss.item()))
            if (step+1) % 100 == 0:
                for param in opt.param_groups:
                    param['lr'] = 1e-3 * (1 - step / 10000.0)

        return fc

    def vectors(self, key):
        if key not in self.buffer["vectors"]:
            path = os.path.join(self.config.vec_path, "{0}_test_{1}.vec".format(key, self.config.pretrained_model))
            real_value_vec, labels = torch.load(path, map_location="cpu")
            real_value_vec = real_value_vec.mean(dim=-1).to(self.config.device)
            if self.config.test_distance == "angular":
                real_value_vec = torch.nn.functional.normalize(real_value_vec, dim=1)
            with torch.no_grad():
                binary_value_vec = (self.dim_reduction_mapping(real_value_vec) > 0).byte()
            self.buffer["vectors"][key] = (binary_value_vec, labels)
        return self.buffer["vectors"][key]

    def compute_distance(self, skt_feat, pho_feat):
        return (skt_feat != pho_feat).byte().sum(dim=1)
