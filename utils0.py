# import os
#
# import numpy as np
# import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
#
#
# def calculate_metric_percase(pred, gt):
#     """
#     计算单个样本的 Dice 和 HD95
#     pred: 预测分割图 (numpy array)
#     gt: 真实标签 (numpy array)
#     """
#     # 二值化
#     pred = (pred > 0).astype(np.bool_)
#     gt = (gt > 0).astype(np.bool_)
#
#     # 情况1: 预测和 GT 都为空 → 完美匹配
#     if not pred.any() and not gt.any():
#         return 1.0, 0.0  # 或 (1.0, float('nan'))
#
#     # 情况2: 一个为空，一个不为空 → 完全不匹配
#     elif not pred.any() or not gt.any():
#         return 0.0, float('inf')  # HD95 无穷大
#
#     # 情况3: 两者都非空 → 正常计算
#     else:
#         dice = metric.binary.dc(pred, gt)
#         try:
#             hd95 = metric.binary.hd95(pred, gt)
#         except:
#             hd95 = float('inf')  # 计算失败也视为无限远
#         return dice, hd95
#
#
# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     if image.dim() == 3:
#         image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
#     if label.dim() == 3:
#         label = label.unsqueeze(0)
#
#     image_np = image.squeeze(0).cpu().numpy()  # [C, H, W]
#     label_np = label.squeeze(0).cpu().numpy()
#     prediction = np.zeros_like(label_np)
#
#     net.eval()
#     with torch.no_grad():
#         for ind in range(image.shape[0]):
#             slice_np = image_np[:, ind] if image_np.shape[0] == 1 else image_np[ind]  # [C, H, W] or [H, W]
#             if len(slice_np.shape) == 2:
#                 slice_np = slice_np[np.newaxis, ...]  # [1, H, W]
#
#             c, h, w = slice_np.shape
#             if h != patch_size[0] or w != patch_size[1]:
#                 slice_np = zoom(slice_np, (1, patch_size[0] / h, patch_size[1] / w), order=3)
#
#             # ✅ 正确转为 tensor: [C, H, W]
#             input_tensor = torch.from_numpy(slice_np).unsqueeze(0).float().cuda()  # [1, C, H, W]
#
#             # 推理
#             output = net(input_tensor)
#             out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
#             out = out.cpu().numpy()
#
#             # 恢复原始尺寸
#             if h != patch_size[0] or w != patch_size[1]:
#                 pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
#             else:
#                 pred = out
#
#             prediction[ind] = pred
#
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label_np == i))
#
#     if test_save_path is not None:
#         os.makedirs(test_save_path, exist_ok=True)
#         img_itk = sitk.GetImageFromArray(image_np.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label_np.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
#         sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
#         sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")
#
#     return metric_list
import os

import numpy as np
from scipy.ndimage import zoom
import torch
import SimpleITK as sitk
# from torchmetrics import metric


# def calculate_metric_percase(pred, gt):
#     """
#     计算单个样本的 Dice 和 HD95
#     pred: 预测分割图 (numpy array)
#     gt: 真实标签 (numpy array)
#     """
#     # 二值化
#     pred = (pred > 0).astype(np.bool_)
#     gt = (gt > 0).astype(np.bool_)
#
#     # 情况1: 预测和 GT 都为空 → 完美匹配
#     if not pred.any() and not gt.any():
#         return 1.0, 0.0  # 或 (1.0, float('nan'))
#
#     # 情况2: 一个为空，一个不为空 → 完全不匹配
#     elif not pred.any() or not gt.any():
#         return 0.0, float('inf')  # HD95 无穷大
#
#     # 情况3: 两者都非空 → 正常计算
#     else:
#         try:
#             dice = metric.binary.dc(pred, gt)
#             hd95 = metric.binary.hd95(pred, gt)
#         except Exception as e:
#             print(f"Error calculating metrics: {e}")
#             dice = 0.0
#             hd95 = float('inf')
#         return dice, hd95





# def calculate_metric_percase(pred_binary, gt_binary):
#     """
#     计算单个类别、单个样本的 Dice 和 HD95
#     pred_binary: 二值 mask (H, W) 或 (D, H, W)，bool 或 0/1，表示某一类（如 class 2）
#     gt_binary:   同上
#     """
#     pred_binary = pred_binary.astype(bool)
#     gt_binary = gt_binary.astype(bool)
#
#     if not pred_binary.any() and not gt_binary.any():
#         return 1.0, 0.0
#     elif not pred_binary.any() or not gt_binary.any():
#         return 0.0, float('inf')
#     else:
#         try:
#             dice = metric.binary.dc(pred_binary, gt_binary)
#             hd95 = metric.binary.hd95(pred_binary, gt_binary)
#             return dice, hd95
#         except Exception as e:
#             print(f"Error in HD95 calculation: {e}")
#             return 0.0, float('inf')
#
# # gai 9.29
# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     """
#     对每个2D切片进行预测并评估
#     返回: list of list of [dice, hd95] -> shape (num_slices, num_classes-1, 2)
#     """
#     if image.dim() == 3:
#         image = image.unsqueeze(0)  # [B, C, H, W] or [B, H, W]
#     if label.dim() == 3:
#         label = label.unsqueeze(0)
#
#     image_np = image.squeeze(0).cpu().numpy()  # [C, D, H, W] or [D, H, W]
#     label_np = label.squeeze(0).cpu().numpy()
#
#     # Determine number of slices (assume depth is the second dim if channel exists)
#     if image_np.ndim == 4:
#         num_slices = image_np.shape[1]  # [C, D, H, W]
#     else:
#         num_slices = image_np.shape[0]  # [D, H, W]
#
#     net.eval()
#     metric_list = []  # Will be: [ [ [d,h], [d,h], ... ], ... ]  -> (num_slices, classes-1, 2)
#
#     for ind in range(num_slices):
#         # Extract slice
#         if image_np.ndim == 4:
#             slice_np = image_np[:, ind]  # [C, H, W]
#         else:
#             slice_np = image_np[ind][np.newaxis, ...]  # [1, H, W]
#
#         c, h, w = slice_np.shape
#         if h != patch_size[0] or w != patch_size[1]:
#             slice_np = zoom(slice_np, (1, patch_size[0] / h, patch_size[1] / w), order=3)
#
#         input_tensor = torch.from_numpy(slice_np).unsqueeze(0).float().cuda()  # [1, C, H, W]
#
#         with torch.no_grad():
#             output = net(input_tensor)
#             out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
#
#             if h != patch_size[0] or w != patch_size[1]:
#                 pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
#             else:
#                 pred = out
#
#         # Get GT slice
#         if label_np.ndim == 4:
#             gt_slice = label_np[:, ind]
#         else:
#             gt_slice = label_np[ind]
#
#         # Compute per-class metrics for this slice
#         slice_metrics = []
#         for i in range(1, classes):  # skip background
#             dice, hd95 = calculate_metric_percase(pred == i, gt_slice == i)
#             slice_metrics.append([dice, hd95])
#
#         metric_list.append(slice_metrics)
#
#         # Optional: save slice results
#         if test_save_path is not None:
#             os.makedirs(test_save_path, exist_ok=True)
#             img_itk = sitk.GetImageFromArray(slice_np.astype(np.float32))
#             prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
#             lab_itk = sitk.GetImageFromArray(gt_slice.astype(np.float32))
#             img_itk.SetSpacing((1, 1, z_spacing))
#             prd_itk.SetSpacing((1, 1, z_spacing))
#             lab_itk.SetSpacing((1, 1, z_spacing))
#             sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_slice_{ind}_pred.nii.gz")
#             sitk.WriteImage(img_itk, f"{test_save_path}/{case}_slice_{ind}_img.nii.gz")
#             sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_slice_{ind}_gt.nii.gz")
#
#     return metric_list


import numpy as np
from scipy.ndimage import zoom
import torch
from medpy.metric import binary as metric

# def calculate_metric_percase(pred, gt):
#     """
#     计算单个样本的 Dice 和 HD95
#     pred: 预测分割图 (numpy array)，已经为某一类别的二值mask
#     gt: 真实标签 (numpy array)，已经为某一类别的二值mask
#     """
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#
#     if not pred.any() and not gt.any():
#         return 1.0, 0.0
#     elif not pred.any() or not gt.any():
#         return 0.0, float('inf')
#     else:
#         try:
#             dice = metric.dc(pred, gt)
#             hd95 = metric.hd95(pred, gt)
#             return dice, hd95
#         except Exception as e:
#             print(f"Error calculating metrics for this case: {e}")
#             return 0.0, float('inf')
from medpy.metric.binary import dc, hd95

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = dc(pred, gt)
        try:
            h = hd95(pred, gt)
            h = min(h, 300.0)
        except:
            h = 300.0
        return dice, h
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, 300.0

# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     """
#     对每个2D切片进行预测并评估
#     返回: numpy array of shape (num_slices, num_classes, 2)
#           metric[s][c] = [dice, hd95] for slice s and class c
#           背景类 (0) 的指标为 [nan, nan]
#     """
#     if image.dim() == 3:
#         image = image.unsqueeze(0)  # [B, C, H, W]
#     if label.dim() == 3:
#         label = label.unsqueeze(0)
#
#     image_np = image.squeeze(0).cpu().numpy()
#     label_np = label.squeeze(0).cpu().numpy()
#
#     # Determine number of slices
#     if image_np.ndim == 4:
#         num_slices = image_np.shape[1]  # [C, D, H, W]
#     else:
#         num_slices = image_np.shape[0]  # [D, H, W]
#
#     net.eval()
#     # 初始化 (num_slices, num_classes, 2) 的数组
#     metric_list = np.zeros((num_slices, classes, 2))
#     # metric_list[:] = np.nan  # 默认为 nan
#
#     for ind in range(num_slices):
#         # Extract slice
#         if image_np.ndim == 4:
#             slice_np = image_np[:, ind]  # [C, H, W]
#         else:
#             slice_np = image_np[ind][np.newaxis, ...]  # [1, H, W]
#
#         c, h, w = slice_np.shape
#         if h != patch_size[0] or w != patch_size[1]:
#             slice_resized = zoom(slice_np, (1, patch_size[0] / h, patch_size[1] / w), order=3)
#         else:
#             slice_resized = slice_np
#
#         input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).float().cuda()
#
#         with torch.no_grad():
#             output = net(input_tensor)
#             out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
#
#             if h != patch_size[0] or w != patch_size[1]:
#                 pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
#             else:
#                 pred = out
#
#         # Get GT slice
#         if label_np.ndim == 4:
#             gt_slice = label_np[:, ind].squeeze(0)
#         else:
#             gt_slice = label_np[ind]
#
#         # Compute per-class metrics for this slice
#         for i in range(1, classes):  # skip background
#             dice, hd95 = calculate_metric_percase(pred == i, gt_slice == i)
#             metric_list[ind, i, 0] = dice
#             metric_list[ind, i, 1] = hd95
#
#     return metric_list  # shape: (num_slices, num_classes, 2)

# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     """
#     对每个2D切片进行预测并评估
#     返回: numpy array of shape (num_slices, num_classes, 2)
#           metric[s][c] = [dice, hd95] for slice s and class c
#           背景类 (0) 的指标为 [nan, nan]
#     """
#     if image.dim() == 3:
#         image = image.unsqueeze(0)  # [B, C, H, W]
#     if label.dim() == 3:
#         label = label.unsqueeze(0)
#
#     image_np = image.squeeze(0).cpu().numpy()
#     label_np = label.squeeze(0).cpu().numpy()
#
#     # Determine number of slices
#     if image_np.ndim == 4:
#         num_slices = image_np.shape[1]  # [C, D, H, W]
#     else:
#         num_slices = image_np.shape[0]  # [D, H, W]
#
#     net.eval()
#     # 初始化 (num_slices, num_classes, 2) 的数组
#     metric_list = np.zeros((num_slices, classes, 2))
#     # metric_list[:] = np.nan  # 默认为 nan
#
#     for ind in range(num_slices):
#         # Extract slice
#         if image_np.ndim == 4:
#             slice_np = image_np[:, ind]  # [C, H, W]
#         else:
#             slice_np = image_np[ind][np.newaxis, ...]  # [1, H, W]
#
#         c, h, w = slice_np.shape
#         if h != patch_size[0] or w != patch_size[1]:
#             slice_resized = zoom(slice_np, (1, patch_size[0] / h, patch_size[1] / w), order=3)
#         else:
#             slice_resized = slice_np
#
#         input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).float().cuda()
#
#         with torch.no_grad():
#             output = net(input_tensor)
#             # 直接使用 argmax 处理 logits 输出
#             out = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
#
#             if h != patch_size[0] or w != patch_size[1]:
#                 pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
#             else:
#                 pred = out
#
#         # Get GT slice
#         if label_np.ndim == 4:
#             gt_slice = label_np[:, ind].squeeze(0)
#         else:
#             gt_slice = label_np[ind]
#
#         # Compute per-class metrics for this slice
#         for i in range(1, classes):  # skip background
#             dice, hd95 = calculate_metric_percase(pred == i, gt_slice == i)
#             metric_list[ind, i, 0] = dice
#             metric_list[ind, i, 1] = hd95
#
#     return metric_list  # shape: (num_slices, num_classes, 2)

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    """
    对每个2D切片进行预测并评估
    返回: numpy array of shape (num_slices, num_classes, 2)
          metric[s][c] = [dice, hd95] for slice s and class c
          背景类 (0) 的指标为 [nan, nan]
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [B, C, D, H, W] -> but B=1
    if label.dim() == 3:
        label = label.unsqueeze(0)

    image_np = image.squeeze(0).cpu().numpy()  # [C, D, H, W] or [D, H, W]
    label_np = label.squeeze(0).cpu().numpy()  # [D, H, W] (integer labels)

    assert image_np.ndim in [3, 4], f"Image must be 3D or 4D, got {image_np.ndim}"
    assert label_np.ndim == 3, f"Label must be 3D (D, H, W), got {label_np.ndim}"

    num_slices = image_np.shape[1] if image_np.ndim == 4 else image_np.shape[0]

    net.eval()
    metric_list = np.full((num_slices, classes, 2), np.nan)  # 初始化为 nan

    device = next(net.parameters()).device

    for ind in range(num_slices):
        # Extract image slice
        if image_np.ndim == 4:
            slice_np = image_np[:, ind]  # [C, H, W]
        else:
            slice_np = image_np[ind][np.newaxis, ...]  # [1, H, W]

        c, h, w = slice_np.shape
        if (h, w) != tuple(patch_size):
            slice_resized = zoom(slice_np, (1, patch_size[0] / h, patch_size[1] / w), order=3)
        else:
            slice_resized = slice_np

        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = net(input_tensor)
            out = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            if (h, w) != tuple(patch_size):
                pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            else:
                pred = out

        # Get GT slice
        gt_slice = label_np[ind]  # [H, W]

        # Compute metrics for non-background classes
        for i in range(1, classes):
            dice, hd95 = calculate_metric_percase(pred == i, gt_slice == i)
            metric_list[ind, i, 0] = dice
            metric_list[ind, i, 1] = hd95

    return metric_list