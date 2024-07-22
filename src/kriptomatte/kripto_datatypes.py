import enum

import Imath
import numpy as np


class ExrDtype(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1


pixel_dtype = {
    ExrDtype.FLOAT32: Imath.PixelType(Imath.PixelType.FLOAT),
    ExrDtype.FLOAT16: Imath.PixelType(Imath.PixelType.HALF),
}
numpy_dtype = {
    ExrDtype.FLOAT32: np.float32,
    ExrDtype.FLOAT16: np.float16,
}
