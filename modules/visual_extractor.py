from medclip import MedCLIPModel, MedCLIPVisionModelViT,MedCLIPVisionModel
from medclip import MedCLIPProcessor
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        # model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        if 0:
             model = models.resnet101()
             model.load_state_dict(torch.load('/root/R2G/model/resnet101/model.pth'))
        else:
            model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
            model.from_pretrained()
            model = model.vision_model.model
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # 一个是总特征，一个是平均特征
        return patch_feats, avg_feats
