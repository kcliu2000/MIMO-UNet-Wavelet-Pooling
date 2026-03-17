'''2026/3/12
import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
'''

import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
import numpy as np

# 新增 structural_similarity 的引入
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    
    psnr_adder = Adder()
    ssim_adder = Adder() # <--- 新增 SSIM 的收集器

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # 將維度從 (C, H, W) 轉換為 (H, W, C) 以符合 skimage SSIM 的標準格式
            p_numpy = np.transpose(p_numpy, (1, 2, 0))
            label_numpy = np.transpose(label_numpy, (1, 2, 0))

            # 計算 PSNR 與 SSIM
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            ssim = structural_similarity(p_numpy, label_numpy, data_range=1, channel_axis=-1)

            psnr_adder(psnr)
            ssim_adder(ssim) # <--- 記錄 SSIM
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    
    # <--- 改為同時回傳 PSNR 與 SSIM 的平均值
    return psnr_adder.average(), ssim_adder.average()