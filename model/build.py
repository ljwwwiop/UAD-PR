from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights

# from model import longclip
# from .model_longclip import build_model
from model import simple_tokenizer

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb
import torch.nn.functional as F


'''
Uncertainty-Aware Disentanglement for Weakly Supervised Text-Based Person Retrieval

Uncertainty-Guided Disentanglement for Weakly Supervised Text-Based Person Retrieval

Probabilistic Disentanglement for Uncertainty-Aware Weakly Supervised Text-Based Person Retrieval
'''

class UAD(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self._set_task()
        self.e_l = args.e_l
        self.margin = args.margin

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']  
        self.tokenizer = simple_tokenizer.SimpleTokenizer()

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.topk = 3 # 

        proj_std = 0.015625000000
        fc_std = 0.03125
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')] 
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, flag, image_pseudo_labels=None, n_iter=None, epoch=None):
        # pdb.set_trace()
        ret = dict()
        images = batch['images']    # 
        caption_ids = batch['caption_ids']   
        image_feats, text_feats = self.base_model(images, caption_ids)  # [64, 193, 512], [64, 77, 512]
       
        i_feats = image_feats[:, 0, :].float()  # for CLIP ViT-B/16 model   
        # i_feats = image_feats.float()         # for CLIP ResNet visual model

        word_feats = text_feats.clone().float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() 

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        # aligned img with word
        topk = self.topk
        topk_indices = torch.full((i_feats.shape[0], topk), -1, dtype=torch.long).to(caption_ids.device)

        # import pdb; pdb.set_trace()

        for i in range(word_feats.shape[0]):
            cur_img = i_feats[i].unsqueeze(0) # [1,512]
            current_length = caption_ids.argmax(dim=-1)[i]
            cur_text = word_feats[i,:current_length,:]  # [S, 512]

            cur_img_norm = torch.nn.functional.normalize(cur_img, p=2, dim=-1)
            cur_text_norm = torch.nn.functional.normalize(cur_text, p=2, dim=-1) 
            cur_sim = cur_img_norm @ cur_text_norm.transpose(0, 1) # [1,77]

            prob = torch.softmax(cur_sim, dim=-1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=0)
            entropy = entropy * entropy
            entropy_weights = torch.exp(-1 * entropy)
            # _, top_idxs = torch.topk(entropy.squeeze(), k=topk, dim=-1, largest=False)
            _, top_idxs = torch.topk(entropy_weights.squeeze(), k=topk, dim=-1, largest=False)

            # _, top_idxs = cur_sim.topk(k=topk)
            topk_indices[i, :topk] = top_idxs
        
        # mask word
        dyn_mask = torch.zeros_like(caption_ids)
        extracted_values = caption_ids.gather(dim=1, index=topk_indices) # [64,1]
        dyn_mask.scatter_(dim=1, index=topk_indices, src=extracted_values) # [64, 77]
        dyn_labels = dyn_mask[(dyn_mask != 0)]

        dyn_feats = image_feats[:,0,:]
        dyn_feats = dyn_feats.unsqueeze(1).repeat(1, topk, 1)
        dyn_predict = self.mlm_head(dyn_feats)
        # scores = dyn_predict[:,0,:].float().reshape(-1, self.args.vocab_size)
        scores = dyn_predict.float().reshape(-1, self.args.vocab_size)

        if flag == True:     # calculate loss
            if 'cdm' in self.current_task:
                ret.update({'cdm_loss':objectives.compute_cdm(i_feats, t_feats, image_pseudo_labels, n_iter, logit_scale)})

            # if epoch > 5:
                # # align loss
                # hard_negative_weight = torch.exp(-cur_sim[top_idxs])  # 越接近图像，权重越高
                # dyn_loss = torch.sum(hard_negative_weight * objectives.compute_mlm(scores, dyn_labels))
                ret.update({'dyn_loss':objectives.compute_mlm(scores, dyn_labels) * 1.0 / topk})

            if epoch > self.e_l:
                if 'chm' in self.current_task:
                    ret.update({'chm_loss':objectives.compute_chm(i_feats, t_feats, image_pseudo_labels, self.margin, n_iter)})    

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
                        
            return ret
        
        else :  # for clustering
            return i_feats
   

def build_model(args, num_classes=11003):
    model = UAD(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
