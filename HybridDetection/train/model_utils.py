import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
import pdb
def get_dino_model(dino_name='facebook/dinov2-base'):
    model = AutoModel.from_pretrained(dino_name)
    model.eval()  
    return model

    
def get_feats_and_meta(dloader, model, device, ignore_feats=False):
    all_feats = None
    labels = []
    imgpaths = []

    for img, lbl, imgpath in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            feats = None
            if not ignore_feats:
                feats = model(img.to(device))[0]
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L707
                cls_token = feats[:, 0]
                patch_tokens = feats[:, 1:]
                feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).cpu().numpy()
                # print('dino feat shape:', feats.shape) #[batchsize,1536]
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats
                
        labels.extend(lbl.cpu().numpy().tolist())
        imgpaths.extend(list(imgpath))
        
    labels = np.array(labels)
    imgpaths = np.array(imgpaths)
    return all_feats, labels, imgpaths
