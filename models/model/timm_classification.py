import timm
from models.model.models import Model


@Model.register("timm_classification")
class TimmClassificationModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        backbone = kwargs["backbone"]
        num_classes = kwargs["num_classes"]
        pretrained = kwargs["pretrained"]
        self.model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, src):
        x = self.model.forward(src)
        return x
