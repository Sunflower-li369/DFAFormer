import numpy as np
import torch
import cv2
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
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
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


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, out_dir = ''):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float().cuda()   # gai # cuda()

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred

            if out_dir != '': # gai 增，，，保存分割结果

                # 配色相关   # gai 增
                alpha = 1.0
                # RGB
                # color_mask = [
                #     np.array([0, 0, 0]),  # 背景 (Background)
                #     np.array([30, 144, 255]),  # 主动脉 (aorta)  blue
                #     np.array([0, 255, 0]),  # 胆囊 (gallbladder)  green
                #     np.array([255, 0, 0]),  # 左肾 (left kidney)  red
                #     np.array([0, 255, 255]),  # 右肾 (right kidney)  cyan
                #     np.array([255, 0, 255]),  # 肝脏 (liver)   pink
                #     np.array([255, 255, 0]),  # 胰腺 (pancreas)  yellow
                #     np.array([128, 0, 255]),  # 脾脏 (spleen)    purple
                #     np.array([200, 200, 200])  # 胃 (stomach)    light gray
                # ]

                # BGR  正确版
                color_mask = [
                    np.array([0, 0, 0]),  # 背景 (Background)
                    np.array([255, 0, 0]),  # 主动脉 (aorta)  blue （B: 255, G: 144, R: 30）
                    np.array([0, 255, 0]),  # 胆囊 (gallbladder)  green （G: 255）
                    np.array([0, 0, 255]),  # 左肾 (left kidney)  red （R: 255）
                    np.array([255, 255, 0]),  # 右肾 (right kidney)  cyan （B: 255, G: 255）
                    np.array([255, 0, 255]),  # 肝脏 (liver)   pink （B: 255, R: 255）
                    np.array([0, 255, 255]),  # 胰腺 (pancreas)  yellow （G: 255, R: 255）
                    np.array([255, 204, 51]),  # 脾脏 (spleen)    蓝
                    np.array([234, 234, 234])  # 胃 (stomach)    light gray
                ]

                # color_mask = [
                #     np.array([0, 0, 0]),  # 背景
                #     np.array([255, 0, 0]),  # 类1
                #     np.array([0, 255, 0]),  # 类2
                #     np.array([0, 0, 255]),  # 类3
                #     np.array([255, 255, 0]),  # 类4
                #     np.array([255, 0, 255]),  # 类5
                #     np.array([0, 255, 255]),  # 类6
                #     np.array([128, 128, 128]),  # 类7
                #     np.array([128, 0, 0]),  # 类8
                #     np.array([0, 128, 0]),  # 类9
                #     np.array([0, 0, 128]),  # 类10
                #     np.array([128, 128, 0]),  # 类11
                #     np.array([128, 0, 128]),  # 类12
                #     np.array([0, 128, 128]),  # 类13
                #     np.array([192, 192, 192]),  # 类14
                #     np.array([255, 165, 0]),  # 类15 (橙色)
                #     np.array([128, 0, 255]),  # 类16 (紫色)
                #     # np.array([255, 192, 203]),  # 类17 (粉红)
                # ]

                seg_img = image[ind, :, :] * 255
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
                seg_lab = label[ind, :, :]
                seg_pred = cv2.resize(pred, (x, y), interpolation=cv2.INTER_LINEAR)

                img_lab, img_pred = seg_img.copy(), seg_img.copy()

                mask_lab, mask_pred = np.zeros((x, y)), np.zeros((x, y))
                for cls in range(1, classes):  # 从1开始循环以跳过背景
                    indexs = np.where(seg_lab[:, :] == cls)
                    mask_lab = np.zeros((x, y), dtype=bool)
                    mask_lab[indexs[0], indexs[1]] = True
                    img_lab[mask_lab] = img_lab[mask_lab] * (1 - alpha) + color_mask[cls] * alpha  # gai 2.5 确保仅对非背景区域操作

                    indexs = np.where(seg_pred[:, :] == cls)
                    mask_pred = np.zeros((x, y), dtype=bool)
                    mask_pred[indexs[0], indexs[1]] = True
                    img_pred[mask_pred] = img_pred[mask_pred] * (1 - alpha) + color_mask[
                        cls] * alpha  # gai 2.5 确保仅对非背景区域操作

                cv2.imwrite('{}/{}_{}_0_org.jpg'.format(out_dir, case, ind), seg_img)
                cv2.imwrite('{}/{}_{}_1_label.jpg'.format(out_dir, case, ind), img_lab)
                cv2.imwrite('{}/{}_{}_2_pred.jpg'.format(out_dir, case, ind), img_pred)

                cv2.imwrite('{}/{}_{}_0_org.jpg'.format(out_dir, case, ind), seg_img)
                cv2.imwrite('{}/{}_{}_1_label.jpg'.format(out_dir, case, ind), img_lab)
                cv2.imwrite('{}/{}_{}_2_pred.jpg'.format(out_dir, case, ind), img_pred)


    else:
        input = torch.from_numpy(image).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

