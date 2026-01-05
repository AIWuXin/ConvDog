from abc import ABC, abstractmethod
from convdog.core.graph import ConvDogGraph
from convdog.utils.logger import logger

class BasePass(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self, dog_graph: ConvDogGraph) -> bool:
        """
        运行优化 Pass
        :param dog_graph: ConvDogGraph 实例
        :return: bool 是否对图进行了实质性的修改
        """
        logger.info(f"正在执行 Pass: [bold magenta]{self.name}[/]", extra={"markup": True})
        modified = self.process(dog_graph)
        if modified:
            # 拓扑结构变了，必须更新索引
            dog_graph.update_indexes()
        return modified

    @abstractmethod
    def process(self, dog_graph: ConvDogGraph) -> bool:
        pass
