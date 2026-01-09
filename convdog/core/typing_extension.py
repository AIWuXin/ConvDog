from enum import Enum


class BackendType(Enum):
    DEFAULT = 0
    QNN = 1


BACKEND_MAP = {
    BackendType.DEFAULT: [
        "CPU",
        "CUDA",
    ],
    BackendType.QNN: ["QNN"]
}
