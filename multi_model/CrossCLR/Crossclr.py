from torch import nn
import torch
import torch.nn.functional as F
import time 
import numpy as np

def cosine_sim(emb1, emb2):
    """compute cosine similarity of two embeddings
    Args:
        emb1 
        emb2 
    Returns:
        float: cosine similarity between (-1, 1)
    """    
    return emb1.mm(emb2.t())
    
class CrossCLR_onlyIntraModality(nn.Module):
    """
    CrossCLR Loss between 2 groups of embeddings - Only Intra Modality alignment
    ICCV 2021
    """
    def __init__(self, temperature=0.03, negative_weight=0.8, logger = None,device="cuda:4"):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.temperature = temperature 
        self.logger = logger
        self.device = device
        self.negative_w = negative_weight # Weight of negative samples logits.
        self.entropy = nn.CrossEntropyLoss()

    def compute_loss(self, logits, mask):
        return - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(device=self.device,non_blocking=True)

    def forward(self, out, y, video_features, text_features):
        """
        Inputs shape (batch, embed_dim)
        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)
        Returns:
        """
        batch_size = video_features.shape[0]

        # Normalize features 
        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)
        
        # Inter-modality alignment
        logits_per_vid = video_features @ text_features.t()
        logits_per_text = text_features @ video_features.t()

        # Intra-modality alignment
        logits_clstr_vid = video_features @ video_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        logits_per_vid /= self.temperature 
        logits_per_text /= self.temperature 
        logits_clstr_vid /= self.temperature 
        logits_clstr_txt /= self.temperature 

        positive_mask = self._get_positive_mask( video_features.shape[0])
        negatives_vid = logits_clstr_vid * positive_mask
        negatives_txt = logits_clstr_txt * positive_mask

        vid_logits = torch.cat([logits_per_vid, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([logits_per_text, self.negative_w * negatives_txt], dim=1)

        diag = np.eye(batch_size)
        mask_vid = torch.from_numpy((diag)).to(self.device)
        mask_txt = torch.from_numpy((diag)).to(self.device)

        mask_neg_v = torch.zeros_like(negatives_vid)
        mask_neg_t = torch.zeros_like(negatives_txt)
        mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
        mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

        loss_i = self.compute_loss(vid_logits, mask_v)
        loss_t = self.compute_loss(txt_logits, mask_t)

        loss1 = self.entropy(out,y.long())
        loss2 = ((loss_i.mean() + loss_t.mean())  )
        return loss1, loss2 + loss1*1e-2


# def test_cross_clr():
#     video_x = torch.randn(32,320).to("cuda:4")
#     text_x = torch.randn(32,320).to("cuda:4")
#     model = CrossCLR_onlyIntraModality(device="cuda:4")
#     print(model(video_x,text_x))
# test_cross_clr()

