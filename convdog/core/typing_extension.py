from enum import Enum

import numpy as np


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


DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16
}
