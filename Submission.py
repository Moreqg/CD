import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import pycocotools.mask as mutils

from src.models import *

data_dir = "../data/test"
train_images_A = sorted(glob.glob(os.path.join(data_dir, "A/*")))
train_images_B = sorted(glob.glob(os.path.join(data_dir, "B/*")))
df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B})
df["uid"] = df.image_file_A.apply(lambda x: int(os.path.basename(x).split(".")[0]))


def get_model(cfg):
    cfg = cfg.copy()
    model = eval(cfg.pop("type"))(**cfg)
    return model

def get_models(names, folds):
    model_infos = [
        dict(
            ckpt = f"./logs/{name}/f{fold}/last.ckpt",
        ) for name in names for fold in folds
    ]
    models = []
    for model_info in model_infos:
        if not os.path.exists(model_info["ckpt"]):
            model_info['ckpt'] = sorted(glob.glob(model_info['ckpt']))[-1]
        stt = torch.load(model_info["ckpt"], map_location = "cpu")
        cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
        stt = {k[6:]: v for k, v in stt["state_dict"].items()}

        model = get_model(cfg)
        model.load_state_dict(stt, strict = True)
        model.eval()
        model.cuda()
        models.append(model)
    return models

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
from torchvision import transforms as T
to_tensor = T.Compose(
    [
        T.ToTensor()
    ]
)
def load(row):
    imgA = cv2.imread(row.image_file_A)
    imgB = cv2.imread(row.image_file_B)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    
    #  fuliye change
    # change the model train method
    image = np.concatenate((imgA, imgB), axis=2)
    image = to_tensor(image)
    return image, None


def predict(row, model, model_paths, img):
    # img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    img = torch.unsqueeze(img, dim=0).cuda()

    with torch.no_grad():
        preds = []
        for model_path in model_paths:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.cuda()
            pred = model(img)
            pred = pred.squeeze().detach().cpu().numpy()
            preds.append(pred)
        pred = sum(preds) / len(preds)  # result's shape is 512*512,  the function of sum is so ??? get sum of one dimension

    return pred


def get_dt(row, pred, img_id, dts):
    # mask = pred.round().astype(np.uint8) # round remove 4 get 5
    mask = pred.copy()
    mask[mask >= 0.05] = 1  # segmentation result
    mask[mask < 0.05] = 0
    mask = np.uint8(mask)
    nc, label = cv2.connectedComponents(mask, connectivity = 8)

    # pixel = 1 are set to one zone, which are the zone of change detection
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode(ann)
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(mutils.area(rle))
            score = float(pred[label == c].mean())
            dts.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]], 
                    "counts": rle["counts"].decode()},
                "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                "image_id": int(img_id), "id": len(dts),
                "score": float(score)
            })

names = [
    "base"
]
folds = [0]

os.system("mkdir -p results")
sub = df
models = get_models(names, folds)
# models = []
# model_paths = [
#     '../model/semantic_change_UnetPlusPlus_efficientnetb0_FFT_Agu_edge_100_fold_0.pth',
#     '../model/semantic_change_UnetPlusPlus_efficientnetb0_FFT_Agu_edge_100_fold_1.pth',
#     '../model/semantic_change_UnetPlusPlus_efficientnetb0_FFT_Agu_edge_100_fold_2.pth',
#     '../model/semantic_change_UnetPlusPlus_efficientnetb0_FFT_Agu_edge_100_fold_3.pth',
#     '../model/semantic_change_UnetPlusPlus_efficientnetb0_FFT_Agu_edge_100_fold_4.pth',
# ]
import segmentation_models_pytorch  as smp
# first model structure only have encoder and decoder, segmentation_head
# second model structure have encoder, decoder and decoder_change segmentation_head  segmentation_head_change
model = smp.UnetPlusPlus(
    encoder_name='efficientnet-b0',
    encoder_weights='imagenet',
    in_channels=6,
    decoder_attention_type='scse',
    classes=1,
    activation='sigmoid'
)
# second model additionally have decoder_change._decoder_. ...  and segmentation_head_chagne
# for model_path in model_paths:
#     model.load_state_dict(torch.load(model_path))
#     models.append(model)
dts = []
output_dir = '../data/test/label'
for idx in tqdm(range(len(sub))):
    row = sub.loc[idx]
    img, mask = load(row)
    pred = predict(row, model, model_paths, img)
    get_dt(row, pred, row.uid, dts) # instance segmentation
    pred[pred>=0.5]=255
    pred[pred<0.5]=0
    image_name = str(row.uid) + '.png'
    image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(image_path, pred)
with open("./results/test.segm.json", "w+") as f:
    json.dump(dts, f)
os.system("zip -9 -r results.zip results")

# observe test images' change results,
# why pred value is so strange