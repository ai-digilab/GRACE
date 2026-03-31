import logging
import os

import numpy as np
import torch
from models.base import BaseLearner
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.inc_net import GRACENet
from utils.toolkit import count_parameters, tensor2numpy

batch_size = 128
num_workers = 2

# Init
init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# Expansion
epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
weight_decay = 2e-4

# Compression
c_epochs = 130
c_lr = 0.1
c_weight_decay = 2e-4
kd_t = 2


class GRACE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = GRACENet(args, False)
        self._snet = None
        self._static_tasks = 0

        logging.info(args)

    @property
    def samples_new_class(self):
        if self.args["dataset"] == "cifar100":
            return 500

        total_samples = 0
        for i in range(self._known_classes, self._total_classes):
            total_samples += self.data_manager.getlen(i)

        new_class_num = self._total_classes - self._known_classes
        return total_samples // new_class_num

    @property
    def samples_old_class(self):
        if self._known_classes == 0:
            return 0
        return self.exemplar_size // self._known_classes

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._cur_conv_num = len(self._network.convnets)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        for conv in self._network.convnets[:-1]:
            for p in conv.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
            self._network_module_ptr = self._network
        self._network.save_checkpoint()

    def train(self):
        self._network.train()
        self._network_module_ptr.convnets[-1].train()
        for conv in self._network_module_ptr.convnets[:-1]:
            conv.eval()

    def train_snet(self):
        self._snet.train()
        self._snet_module_ptr.convnets[-1].train()
        for conv in self._snet_module_ptr.convnets[:-1]:
            conv.eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            if os.path.exists(self.args["pretrain_path"]):
                checkpoint = torch.load(self.args["pretrain_path"], map_location="cpu")
                self._network.fc.load_state_dict(checkpoint["fc_state_dict"])
                self._network.convnets[0].load_state_dict(
                    checkpoint["convnets_state_dicts"][0]
                )
            else:
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    momentum=0.9,
                    lr=init_lr,
                    weight_decay=init_weight_decay,
                )
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
                )
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._network_module_ptr.weight_align(
                self._total_classes - self._known_classes
            )

            # Compression
            if self._compress:
                self._feature_compression(train_loader, test_loader)
                self._snet.weight_align(self._total_classes - self._known_classes)
                self._cur_conv_num -= 1

                self._snet.reset_com_predictor()
                self._network = self._snet
                self._static_tasks += 1
            else:
                self._static_tasks = 0

        # Check saturation
        saturation = erank_saturation(self._network, train_loader, self._device)
        threshold = self.args["t_base"] * (self.args["t_decay"] ** self._static_tasks)
        self._compress = saturation < threshold
        logging.info(
            f"TASK {self._cur_task + 1}; Compress: {self._compress}; Sat: {saturation}"
        )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        fuse_convnet = len(self._network.convnets) - 2

        # Calculate bias correction factor
        samples_old = self.samples_old_class
        samples_new = self.samples_new_class
        bias_factor = samples_new / (samples_new + samples_old)
        logging.info(f"Bias correction factor: {bias_factor}")

        self._snet = GRACENet(self.args, False)
        self._snet.init_student(self._network_module_ptr, fuse_convnet, bias_factor)

        for conv in self._snet.convnets[:-1]:
            for p in conv.parameters():
                p.requires_grad = False
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=c_lr,
            momentum=0.9,
            weight_decay=c_weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=c_epochs
        )

        kd_lambda = self._known_classes / self._total_classes

        self._network.eval()
        prog_bar = tqdm(range(c_epochs))
        for _, epoch in enumerate(prog_bar):
            self.train_snet()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                s_out = self._snet(inputs)
                with torch.no_grad():
                    t_out = self._network(inputs)

                # Predictor loss
                com_feature = self._get_com_feature(t_out, fuse_convnet)
                loss_pred = F.mse_loss(com_feature, s_out["com_pred_feature"])

                # Classification loss
                loss_clf = F.cross_entropy(s_out["logits"], targets)

                # Knowledge distillation loss
                loss_kd = _KD_loss(s_out["logits"], t_out["logits"], kd_t)

                # Final loss
                loss = (1 - kd_lambda) * loss_clf + kd_lambda * (
                    self.args["c_kd_factor"] * loss_kd
                    + self.args["c_pred_factor"] * loss_pred
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(s_out["logits"], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    c_epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    c_epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
        logging.info("Params before: {}".format(count_parameters(self._network)))
        logging.info("Params after: {}".format(count_parameters(self._snet)))

        if isinstance(self._snet, nn.DataParallel):
            self._snet = self._snet.module
            self._snet_module_ptr = self._snet

    def _get_com_feature(self, t_out, fuse_convnet):
        feats = t_out["features"]
        out_dim = self._network.out_dim

        start = fuse_convnet * out_dim
        end = (fuse_convnet + 1) * out_dim
        conv1_feat = feats[:, start:end]
        conv2_feat = feats[:, -out_dim:]

        return torch.cat([conv1_feat, conv2_feat], 1)


def erank_saturation(model, loader, device):
    """
    Calculates the Normalized Effective Rank (ER) of the model's feature space.

    Returns:
        float: Saturation ratio between 0.0 (collapse) and 1.0 (full capacity utilization).
    """
    model.eval()
    model.to(device)

    features_list = []

    def hook_fn(module, input, output):
        # Flatten and move to CPU to avoid VRAM overflow during accumulation
        features_list.append(output.detach().cpu().flatten(start_dim=1))

    handle = model.convnets[-1].avgpool.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(device)
                _ = model(inputs)
    finally:
        handle.remove()

    if not features_list:
        return 0.0

    # Stack and center features
    Z = torch.cat(features_list, dim=0)
    Z = Z - Z.mean(dim=0, keepdim=True)

    # Compute Singular Values
    if hasattr(torch.linalg, "svdvals"):
        S = torch.linalg.svdvals(Z)
    else:
        _, S, _ = torch.svd(Z)

    # Filter out numerical noise
    S = S[S > 1e-5]
    if len(S) == 0:
        return 0.0

    # Calculate Effective Rank (Shannon Entropy of Singular Values)
    p = S / S.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    erank = torch.exp(entropy).item()

    # Normalize by theoretical maximum capacity (min of samples or neurons)
    max_rank = min(Z.shape)
    if max_rank > 0:
        return erank / max_rank
    return 0.0


def _KD_loss(s_logits, t_logits, T):
    p_student = F.log_softmax(s_logits / T, dim=1)
    p_teacher = F.softmax(t_logits / T, dim=1)
    return F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)
