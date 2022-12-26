from partial_fc import PartialFC, PartialFCAdamW
from backbones import get_model
import logging
from torchviz import make_dot
import torch
from utils.utils_config import get_config
from losses import CombinedMarginLoss
cfg=get_config(r"configs/glint360k_r50")

margin_loss = CombinedMarginLoss(
    64,
    cfg.margin_list[0],
    cfg.margin_list[1],
    cfg.margin_list[2],
    cfg.interclass_filtering_threshold
)

backbone = get_model(
    cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
backbone.eval()
torch.cuda.set_device(0)


x = (torch.rand(1,3,112,112)-0.5)/0.5
x=x.cuda()
y = backbone(x)

if __name__=="__main__":
    dot=make_dot(y.mean(), params=dict(backbone.named_parameters()))
    dot.format = 'svg'
    dot.render()