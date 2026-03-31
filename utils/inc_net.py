import copy
import logging
import os
from datetime import datetime

import torch
from convs.cifar_resnet import resnet32
from convs.linears import SimpleLinear
from convs.resnet import resnet18, resnet34, resnet50
from torch import nn


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained, args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class GRACENet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        # Compression Related
        self.conv_classes = []
        self.com_predictor = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def init_student(self, teacher_net, fuse_convnet, bias_factor):
        # Copy teacher task sizes
        self.task_sizes = teacher_net.task_sizes
        self.conv_classes = teacher_net.conv_classes

        # Extract fused convnet learnt classes
        t_conv1_classes = self.conv_classes[fuse_convnet]
        t_conv2_classes = self.conv_classes[-1]

        # Recalculate classes per convnet
        fuse_conv_classes = self.conv_classes.pop(fuse_convnet)
        self.conv_classes[-1] += fuse_conv_classes

        # Copy all kept convnets
        for i, convnet in enumerate(teacher_net.convnets[:-1]):
            if i != fuse_convnet:
                self.convnets.append(copy.deepcopy(convnet))

        # Create new convnet and initialize with teacher weights
        self.convnets.append(get_convnet(self.args))
        pres_factor = t_conv1_classes / (t_conv1_classes + t_conv2_classes)
        p = self.args["preservation_exponent"]
        weight_factor = ((pres_factor**p + bias_factor**p) / 2) ** (1 / p)
        logging.info(f"Previous weight factor: {weight_factor}")

        for new, t1, t2 in zip(
            self.convnets[-1].parameters(),
            teacher_net.convnets[fuse_convnet].parameters(),
            teacher_net.convnets[-1].parameters(),
        ):
            new.data = weight_factor * t1.data + (1 - weight_factor) * t2.data

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        # Create new FC
        nb_classes = teacher_net.fc.out_features
        self.fc = self.generate_fc(self.feature_dim, nb_classes)

        # Initialize feature predictor
        self.com_predictor = self.generate_fc(self.out_dim, 2 * self.out_dim)

    def reset_com_predictor(self):
        self.com_predictor = None

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def extract_features(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        return features

    def forward(self, x):
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        out = self.fc(features)

        if self.aux_fc:
            aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]
        else:
            aux_logits = None

        out.update({"aux_logits": aux_logits, "features": features})

        # Compression feature prediction
        if self.com_predictor is not None:
            com_pred_feature = self.com_predictor(features[:, -self.out_dim :])[
                "logits"
            ]
            out.update(com_pred_feature=com_pred_feature)

        return out

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.conv_classes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        logging.info(f"Alignweights gamma: {gamma}")
        self.fc.weight.data[-increment:, :] *= gamma

    def save_checkpoint(self):
        path = self.args.get("save_path", None)
        if path is None:
            return None

        if "save_folder" not in self.args:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            folder_name = f"{self.args['prefix']}_{timestamp}"
            self.args["save_folder"] = os.path.join(path, folder_name)
            os.makedirs(self.args["save_folder"], exist_ok=True)

        task_num = len(self.task_sizes) - 1
        filename = f"task_{task_num}.pt"
        full_path = os.path.join(self.args["save_folder"], filename)

        checkpoint = {
            "args": self.args,
            "fc_in": self.fc.in_features,
            "fc_out": self.fc.out_features,
            "fc_state_dict": self.fc.state_dict(),
            "convnets_state_dicts": [conv.state_dict() for conv in self.convnets],
            "task_sizes": self.task_sizes,
            "conv_classes": self.conv_classes,
        }

        torch.save(checkpoint, full_path)
        logging.info(f"Checkpoint saved at: {full_path}")
        return full_path

    @staticmethod
    def load_checkpoint(path):
        checkpoint = torch.load(path, map_location="cpu")
        model = GRACENet(checkpoint["args"], True)

        fc_in = checkpoint["fc_in"]
        fc_out = checkpoint["fc_out"]
        model.fc = model.generate_fc(fc_in, fc_out)
        model.fc.load_state_dict(checkpoint["fc_state_dict"])

        for conv_state in checkpoint["convnets_state_dicts"]:
            conv = get_convnet(checkpoint["args"])
            conv.load_state_dict(conv_state)
            model.convnets.append(conv)

        model.task_sizes = checkpoint["task_sizes"]
        model.conv_classes = checkpoint["conv_classes"]

        return model
