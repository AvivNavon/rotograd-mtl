import logging
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import trange

import rotograd
from rotograd import RotoGrad

from nyuv2.data import NYUv2
from nyuv2.utils import (
    set_seed,
    set_logger,
    get_device,
    str2bool,
    ConfMatrix,
    depth_error,
    normal_error
)
from nyuv2.models import SegnetBackbone, SegmentationHead, DepthHead, NormalHead

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    kernels = 64
    dim = 1024
    backbone = SegnetBackbone().to(device)
    heads = [SegmentationHead(kernels).to(device), DepthHead(kernels).to(device), NormalHead(kernels).to(device)]
    model = RotoGrad(backbone, heads, dim, normalize_losses=True).to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # optimizer
    if args.optim == "sgd":
        optim_model = torch.optim.SGD(
            [{'params': m.parameters()} for m in [backbone] + heads], lr=lr, momentum=0.9
        )
        optim_rotograd = torch.optim.SGD(model.parameters(), lr=args.method_params_lr)

    else:
        optim_model = torch.optim.Adam(
            [{'params': m.parameters()} for m in [backbone] + heads], lr=lr,
        )
        optim_rotograd = torch.optim.Adam(model.parameters(), lr=args.method_params_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optim_model, step_size=100, gamma=0.5)
    scheduler_rotograd = torch.optim.lr_scheduler.StepLR(optim_rotograd, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(13)
    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()

            optim_model.zero_grad()
            optim_rotograd.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            with rotograd.cached():
                train_pred = model(train_data)
                losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                    )
                )
                model.backward(losses, retain_graph=True)
                optim_model.step()
                optim_rotograd.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

        # scheduler
        scheduler.step()
        scheduler_rotograd.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function - add eval every?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(13)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30"
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} ||"
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f}"
            )


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2 - Rotograd")
    parser.add_argument(
        "--data-path",
        type=str,
        help="path to data",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--method-params-lr",
        type=float,
        default=5e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="num. epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="batch size",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)
