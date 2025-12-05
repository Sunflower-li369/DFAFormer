import logging
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils0 import test_single_volume


#
# # Define the device (GPU or CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = 9   # gai zeng
#
# def load_model(model_path, num_classes):
#     """
#     Load the pre-trained model.
#     """
#     from networks.DAEFormer_DIFF_decoder_SGFN import DAEFormer  # Import your model architecture
#     model = DAEFormer(num_classes=num_classes).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     return model
#
# def preprocess_image(image_path, img_size):
#     """
#     Preprocess the input image for inference.
#     """
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#     return image_tensor
#
# def inference_single_image(model, image_path, img_size, num_classes):
#     """
#     Perform inference on a single image and return the segmentation result.
#     """
#     # Preprocess the input image
#     image_tensor = preprocess_image(image_path, img_size)
#
#     # Perform inference
#     with torch.no_grad():
#         output = model(image_tensor)
#         pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Get segmentation mask
#
#     return pred
#
# def postprocess_segmentation(pred, img_size):
#     """
#     Postprocess the segmentation mask to an image format.
#     """
#     from PIL import Image
#     segmentation_image = Image.fromarray((pred * 255 / num_classes).astype(np.uint8))
#     return segmentation_image

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(
        base_dir=args.volume_path,
        split="test_vol",
        img_size=args.img_size,
        list_dir=args.list_dir
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    all_case_metrics = []  # Will store (num_classes-1, 2) per case

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        case_name = str(case_name)

        # Returns: list of shape (num_slices, num_classes-1, 2)
        metric_list = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )

        # Convert to numpy array: (num_slices, num_classes-1, 2)
        metric_array = np.array(metric_list)  # e.g., (150, 8, 2)

        # Average over all slices to get per-case per-class metrics
        case_metric = np.mean(metric_array, axis=0)  # shape: (8, 2)

        all_case_metrics.append(case_metric)

        # Extract Dice and HD95 across all classes
        dices = case_metric[:, 0]    # (8,)
        hd95s = case_metric[:, 1]    # (8,)

        mean_dice = float(np.mean(dices))
        mean_hd95 = float(np.mean(hd95s))

        logging.info("idx %d case %s mean_dice %f mean_hd95 %f" % (
            i_batch, case_name, mean_dice, mean_hd95
        ))

    # Stack all cases: (num_cases, num_classes-1, 2)
    all_metrics = np.stack(all_case_metrics, axis=0)  # e.g., (12, 8, 2)
    num_fg = args.num_classes - 1

    # Per-class statistics
    for i in range(num_fg):
        dice_scores = all_metrics[:, i, 0]
        hd95_scores = all_metrics[:, i, 1]
        logging.info(
            "Class %d: Dice = %.4f ± %.4f, HD95 = %.4f ± %.4f"
            % (i + 1, float(np.mean(dice_scores)), float(np.std(dice_scores)),
               float(np.mean(hd95_scores)), float(np.std(hd95_scores)))
        )

    # Overall: mean over classes, then over cases
    overall_dice_per_case = np.mean(all_metrics[:, :, 0], axis=1)  # (num_cases,)
    overall_hd95_per_case = np.mean(all_metrics[:, :, 1], axis=1)

    logging.info(
        "Overall Mean Dice: %.4f ± %.4f, Mean HD95: %.4f ± %.4f"
        % (float(np.mean(overall_dice_per_case)), float(np.std(overall_dice_per_case)),
           float(np.mean(overall_hd95_per_case)), float(np.std(overall_hd95_per_case)))
    )

    return "Testing Finished!"