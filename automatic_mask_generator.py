from re import T
from IPython.display import display, HTML
from sympy import imageset
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def get_img_SAM(in_img_path, out_img_path):
    image = cv2.imread(in_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)


    w,h = 300,300
    dpi = 300
    fig = plt.figure(figsize=(1.3, 1.3), dpi=dpi)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    fig.savefig(out_img_path, bbox_inches='tight',pad_inches=0.0, dpi=dpi)
    plt.close()

import os
import glob
in_set_list = ["t1"]
mode_list = ["train"]
img_set = 'c-500-set-easy'
# 画像セットのパス

from tqdm import tqdm
for mode in tqdm(mode_list):
    for set in tqdm(in_set_list):
        set_path = f'img_set/{img_set}/set0/{mode}/{set}'
        os.makedirs(f'{set_path}_sam', exist_ok=True)
        #pngファイルのパスを取得
        imgs = glob.glob(f'{set_path}/*.png')
        imgs_already = glob.glob(f'{set_path}_sam/*.png')
        # '_sam' サフィックスを取り除いたバージョンのファイル名を取得
        imgs_already_base = [os.path.basename(img).replace("_sam", "") for img in imgs_already]

        # imgs_already_base にないファイルだけを残す
        imgs_remained = [img for img in imgs if os.path.basename(img) not in imgs_already_base]
        #ファイル名を取得
        for img in tqdm(imgs_remained):
            #ファイル名を取得
            img_name = os.path.basename(img)
            in_path = img
            out_path = f"{set_path}_sam/{img_name.replace('.png', '_sam.png')}"
            get_img_SAM(in_path, out_path)
