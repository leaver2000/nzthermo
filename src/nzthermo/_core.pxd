# pyright: reportGeneralTypeIssues=false

cdef extern from *:
    """
    #ifdef _OPENMP
    #define OPENMP 1
    #else
    #define OPENMP 0
    #endif /* OPENMP */
    """
    cdef bint OPENMP


cdef extern from "<math.h>" nogil:
    bint isnan(long double x)
    const float NaN "NAN"


ctypedef fused T:
    float
    double


ctypedef fused integer:
    short
    long


cdef enum BroadcastMode:
    BROADCAST = 1
    MATRIX = 2
    ELEMENT_WISE = 3

cdef enum ProfileStrategy:
    SURFACE_BASED = 0
    MOST_UNSTABLE = 1
    MIXED_LAYER = 2
    EQUILIBRIUM = 3
    VIRTUAL = 4
