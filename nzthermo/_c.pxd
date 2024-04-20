cdef extern from "<math.h>" nogil:
    double exp(double x)
    double log(double x)
    double ceil(double x)
    double sin(double x)
    double cos(double x)
    double tan(double x)
    double asin(double x)
    double acos(double x)
    double atan(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    bint isnan(long double x)
    const double pi "M_PI"  # as in Python's math module


ctypedef fused floating:
    float
    double


cdef enum BroadcastMode:
    INFER = 0
    BROADCAST = 1
    MATRIX = 2
    SLICE = 3
