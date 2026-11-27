
import numpy as np

# metpy 1.6.2 (the pinned reference for the regression tests) calls ``np.trapz``, which numpy
# removed in favour of ``np.trapezoid``. Patch it here, for the tests only, rather than in nzthermo.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

