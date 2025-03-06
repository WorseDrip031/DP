import timm
import torch.nn as nn
import torchvision.transforms as TVF
import torchvision.models as models

    

class ResNet18Model(nn.Module):
    def __init__(self, num_classes, use_pretrained):
        super(ResNet18Model, self).__init__()
        self.num_classes = num_classes

        self.model = models.resnet18(pretrained=use_pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
    

class ViTModel(nn.Module):
    def __init__(self, num_classes, use_pretrained=True, use_frozen=True):
        super(ViTModel, self).__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained

        # Create model
        self.feature_extractor = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=use_pretrained)
        self.feature_extractor.reset_classifier(0, "")
        self.head = nn.Sequential(
            nn.Linear(self.feature_extractor.num_features, num_classes)
        )

        # Freeze parameters
        if use_frozen:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # Normalization transform
        if use_pretrained:
            self.norm_transform = TVF.Normalize(
                mean=self.feature_extractor.default_cfg["mean"],
                std=self.feature_extractor.default_cfg["std"]
            )


    def forward(self, x):
        if self.use_pretrained:
            x = self.norm_transform(x)
        f = self.feature_extractor(x)
        f = f[:,0]
        logits = self.head(f)
        return logits





class EVA02Model(nn.Module):
    def __init__(self, num_classes, use_pretrained=True, use_frozen=True):
        super(EVA02Model, self).__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained

        # Create model
        self.feature_extractor = timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=use_pretrained)
        self.feature_extractor.reset_classifier(0, "")
        self.head = nn.Sequential(
            nn.Linear(self.feature_extractor.num_features, num_classes)
        )

        # Freeze parameters
        if use_frozen:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # Normalization transform
        if use_pretrained:
            self.norm_transform = TVF.Normalize(
                mean=self.feature_extractor.default_cfg["mean"],
                std=self.feature_extractor.default_cfg["std"]
            )


    def forward(self, x):
        if self.use_pretrained:
            x = self.norm_transform(x)
        f = self.feature_extractor(x)
        f = f[:,0]
        logits = self.head(f)
        return logits