import skimage.feature
import skimage.segmentation
import numpy as np
import torch


def LBP(batch_image):
    batch_lbp_images = []
    
    for image in batch_image:
        image = torch.squeeze(image)
        lbp_images = []
        for slice in image:
            image_cpu = slice.cpu().numpy()
            LBP_img = skimage.feature.local_binary_pattern(image_cpu, 16, 0.1, method='uniform')
            lbp_image_tensor = torch.from_numpy(LBP_img).float().to(image.device)
            lbp_images.append(lbp_image_tensor)
        stacked_lbp_images = torch.stack(lbp_images, dim=0)
        batch_lbp_images.append(stacked_lbp_images)
    processed_batch_data = torch.stack(batch_lbp_images)
    processed_batch_data = torch.unsqueeze(processed_batch_data, dim=1)
    # processed_batch_data=processed_batch_data.double()
    res = torch.tensor(processed_batch_data, requires_grad=True)
    # 返回批次中的LBP图像
    return res
