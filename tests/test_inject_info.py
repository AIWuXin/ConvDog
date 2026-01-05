from convdog.utils.logger import logger
from convdog.core.graph import ConvDogModel


def main():
    logger.debug("启动调试模式...")
    try:
        dog = ConvDogModel("tests/res/onnx/dpts.onnx")
        dog.inject_convdog_info("LethalDog")
        dog.save("tests/res/onnx/dpts_sim.onnx")
    except Exception as e:
        logger.critical(f"紧急故障: {e}")


if __name__ == '__main__':
    main()
