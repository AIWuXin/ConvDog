from abc import ABC, abstractmethod
from convdog.core.graph import ConvDogModel
from convdog.utils.logger import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx_graphsurgeon as gs


class BasePass(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self, dog_graph: ConvDogModel) -> bool:
        """
        运行优化 Pass
        :param dog_graph: ConvDogGraph 实例
        :return: bool 是否对图进行了实质性的修改
        """
        logger.debug(f"正在执行 Pass: [bold magenta]{self.name}[/]", extra={"markup": True})
        modified = self.process(dog_graph.graph)
        dog_graph.sync_model()
        return modified

    @abstractmethod
    def process(self, dog_graph: "gs.Graph") -> bool:
        pass
