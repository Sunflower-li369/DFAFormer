import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils1 import DiceLoss
from torchvision import transforms
from utils0 import test_single_volume
from torch.nn import functional as F
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

import matplotlib.pyplot as plt
import pandas as pd
import datetime

# def inference(model, testloader, args, test_save_path=None):
#     model.eval()
#     metric_list = 0.0
#
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#         logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#
#     metric_list = metric_list / len(testloader.dataset)
#
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#
#     return performance, mean_hd95
def inference(model, testloader, args, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    case_metrics = []  # each element: (num_classes-1, 2) for multi-class, or (2,) for binary

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(
                image, label, model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name,
                z_spacing=args.z_spacing
            )
            metric_i = np.array(metric_i)  # shape: (N, C-1, 2) or (N, 2)

            # Average over slices to get case-level metric
            if metric_i.ndim == 3:
                # multi-class: (N, C-1, 2) -> (C-1, 2)
                case_mean = np.mean(metric_i, axis=0)
            elif metric_i.ndim == 2:
                # binary: (N, 2) -> (2,)
                case_mean = np.mean(metric_i, axis=0)
            else:
                raise ValueError(f"Unexpected metric_i shape: {metric_i.shape}")

            case_metrics.append(case_mean)

            # Log per-case result
            if args.num_classes == 2:
                # case_mean is (2,)
                logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                    i_batch, case_name, case_mean[0], case_mean[1]))
            else:
                # case_mean is (C-1, 2)
                logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                    i_batch, case_name, np.mean(case_mean[:, 0]), np.mean(case_mean[:, 1])))

    case_metrics = np.array(case_metrics)

    if args.num_classes == 2:
        # case_metrics shape: (num_cases, 2)
        mean_dice = np.mean(case_metrics[:, 0])
        mean_hd95 = np.mean(case_metrics[:, 1])
        logging.info('Mean class 1 mean_dice %f mean_hd95 %f' % (mean_dice, mean_hd95))
    else:
        # case_metrics shape: (num_cases, C-1, 2)
        for i in range(1, args.num_classes):
            dice_i = np.mean(case_metrics[:, i-1, 0])
            hd95_i = np.mean(case_metrics[:, i-1, 1])
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, dice_i, hd95_i))
        mean_dice = np.mean(case_metrics[:, :, 0])
        mean_hd95 = np.mean(case_metrics[:, :, 1])

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (mean_dice, mean_hd95))
    logging.info("Testing Finished!")
    return mean_dice, mean_hd95

def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')


# def trainer_synapse(args, model, snapshot_path):
#     n_gpu = args.n_gpu  # gai 12.23 gpu
#
#     os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
#     test_save_path = os.path.join(snapshot_path, 'test')
#
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * n_gpu  # gai 12.23 gpu
#     # max_iterations = args.max_iterations
#
#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     y_transforms = transforms.ToTensor()
#
#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
#                                norm_x_transform=x_transforms, norm_y_transform=y_transforms)
#     # full_resolution=512  # Synapse 数据集原始是 512x512
#     # )   # gai 10.7
#
#     print("The length of train set is: {}".format(len(db_train)))
#
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
#
#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
#                              pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#
#     db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     if n_gpu > 1:
#         model = nn.DataParallel(model)  # gai 12.23 gpu
#
#
#     model.train()
#
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     dice_ = []
#     hd95_ = []
#
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#
#             # Move data to the same device as the model
#             image_batch, label_batch = image_batch.to(device), label_batch.squeeze(1).to(device)
#
#             # Forward pass
#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.4 * loss_ce + 0.6 * loss_dice
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Adjust learning rate
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_
#
#             iter_num += 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#             writer.add_scalar('info/loss_dice', loss_dice, iter_num)
#
#             logging.info(
#                 f'Iteration {iter_num}: loss={loss.item()}, loss_ce={loss_ce.item()}, loss_dice={loss_dice.item()}')
#
#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)
#
#         # Test
#         eval_interval = args.eval_interval
#         if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_mode_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#
#             logging.info("*" * 20)
#             logging.info(f"Running Inference after epoch {epoch_num}")
#             print(f"Epoch {epoch_num}")
#             mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#             dice_.append(mean_dice)
#             hd95_.append(mean_hd95)
#             model.train()
#
#         if epoch_num >= max_epoch - 1:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_mode_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#
#             if not (epoch_num + 1) % args.eval_interval == 0:
#                 logging.info("*" * 20)
#                 logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
#                 print(f"Epoch {epoch_num}, Last Epcoh")
#                 mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#                 dice_.append(mean_dice)
#                 hd95_.append(mean_hd95)
#                 model.train()
#
#             iterator.close()
#             break
#
#     plot_result(dice_, hd95_, snapshot_path, args)
#     writer.close()
#     return "Training Finished!"
# gai 10.11
def trainer_synapse(args, model, snapshot_path):
    n_gpu = args.n_gpu

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * n_gpu

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    # ✅ 关键修改：传入 use_patch
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size,
        use_patch=getattr(args, 'use_patch', False),  # 安全获取，避免 AttributeError
        norm_x_transform=x_transforms,
        norm_y_transform=y_transforms
    )

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # 验证/测试集：通常不用 patch（use_patch=False）
    db_test = Synapse_dataset(
        base_dir=args.test_path,
        split="test_vol",
        list_dir=args.list_dir,
        img_size=args.img_size,
        use_patch=False  # 测试用滑动窗口或全图
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_ = []
    hd95_ = []

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.squeeze(1).to(device)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                f'Iteration {iter_num}: loss={loss.item()}, loss_ce={loss_ce.item()}, loss_dice={loss_dice.item()}')

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Evaluation
        eval_interval = args.eval_interval
        if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()

            iterator.close()
            break

    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    return "Training Finished!"


# In trainer.py (or wherever your trainer is defined)



# gai 10.11: 新增仅验证函数，用于 CV 的验证阶段
def validate_on_volume_list(model, val_loader, args):
    model.eval()
    case_metrics = []

    with torch.no_grad():
        for sampled_batch in val_loader:
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(
                image, label, model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                z_spacing=args.z_spacing
            )
            metric_i = np.array(metric_i)
            if metric_i.ndim == 3:
                case_mean = np.mean(metric_i, axis=0)  # (C-1, 2)
            else:
                case_mean = np.mean(metric_i, axis=0)  # (2,)
            case_metrics.append(case_mean)

    case_metrics = np.array(case_metrics)
    if args.num_classes == 2:
        mean_dice = np.mean(case_metrics[:, 0])
    else:
        mean_dice = np.mean(case_metrics[:, :, 0])  # average over classes and cases
    return mean_dice

# import numpy as np
# import torch
#
# def validate_on_volume_list(model, val_loader, args):
#     model.eval()
#     all_dices = []   # list of (num_classes-1,) arrays
#     all_hd95s = []   # list of (num_classes-1,) arrays
#
#     with torch.no_grad():
#         for sampled_batch in val_loader:
#             image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#             metric_i = test_single_volume(
#                 image, label, model,
#                 classes=args.num_classes,
#                 patch_size=[args.img_size, args.img_size],
#                 z_spacing=args.z_spacing
#             )
#             # metric_i is list of length 2*(num_classes-1): [dice1, dice2, ..., hd951, hd952, ...]
#             metric_i = np.array(metric_i)
#             num_foreground = args.num_classes - 1
#             dices = metric_i[:num_foreground]
#             hd95s = metric_i[num_foreground:]
#
#             all_dices.append(dices)
#             all_hd95s.append(hd95s)
#
#     all_dices = np.array(all_dices)  # shape: (N_cases, num_foreground)
#     all_hd95s = np.array(all_hd95s)  # shape: (N_cases, num_foreground)
#
#     print("\n=== Validation Results (mean ± std) ===")
#     for i in range(all_dices.shape[1]):
#         dice_mean = np.mean(all_dices[:, i])
#         dice_std = np.std(all_dices[:, i])
#         hd95_mean = np.mean(all_hd95s[:, i])
#         hd95_std = np.std(all_hd95s[:, i])
#         print(f"Class {i+1}: Dice = {dice_mean:.4f} ± {dice_std:.4f}, HD95 = {hd95_mean:.2f} ± {hd95_std:.2f}")
#
#     # Overall mean across all classes and cases
#     overall_dice_mean = np.mean(all_dices)
#     overall_dice_std = np.std(all_dices)
#     overall_hd95_mean = np.mean(all_hd95s)
#     overall_hd95_std = np.std(all_hd95s)
#     print(f"Overall: Dice = {overall_dice_mean:.4f} ± {overall_dice_std:.4f}, HD95 = {overall_hd95_mean:.2f} ± {overall_hd95_std:.2f}")
#     print("=" * 50)
#
#     # Return overall mean Dice (for early stopping or best model selection)
#     return overall_dice_mean