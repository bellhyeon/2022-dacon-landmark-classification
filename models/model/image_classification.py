from torch import nn
import timm
from models.model.models import Model


# pylint: disable=invalid-name
@Model.register("image_classification")
class ImageClassificationModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        backbone = kwargs["backbone"]
        fc = kwargs["fc"]
        num_classes = kwargs["num_classes"]

        self.model = timm.create_model(backbone, pretrained=False)
        self.fc = nn.Linear(fc, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, src, tgt=None):
        x = self.model.forward_features(src)
        x = self.gap(x)
        x = x.view([-1, x.shape[1]])
        x = nn.ReLU()(self.fc(x))
        x = nn.ReLU()(self.fc2(x))
        return x
