from typing import List, Any
from torch import nn, Tensor
from abc import abstractmethod


class BaseVae(nn.Module):

    def __int__(self) -> None:
        super(BaseVae, self).__init__()

    def encode(self, input_data: Tensor, data_label: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input_data: Tensor, data_label: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data: Tensor, data_label: Tensor):
        pass

    @abstractmethod
    def loss_function(self, *args: Any, **kwargs):
        pass
