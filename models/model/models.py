from abc import ABCMeta, abstractmethod
from typing import Dict, Any

from torch import nn

_RESISTER: Dict = {}


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()

    @abstractmethod
    def forward(self, src, tgt):
        pass

    @classmethod
    def register(cls, model_name: str):
        def add_model_class(model_class: Any):
            if model_name not in _RESISTER:
                _RESISTER[model_name] = model_class
            return model_class

        return add_model_class

    @classmethod
    def get_model(cls, model_name: str, args):
        if model_name not in _RESISTER:
            raise KeyError(f"{model_name} is not registered in the register.")
        return _RESISTER[model_name](**args)
