"""
Thanks to Julius on
https://stackoverflow.com/questions/33072604/how-to-get-scipy-integrate-odeint-to-stop-when-path-is-closed
"""
import numpy as np
cimport numpy as np
import cython
#cython: boundscheck=False
#cython: wraparound=False

cdef double a2  =   2.500000000000000e-01  #  1/4
cdef double a3  =   3.750000000000000e-01  #  3/8
cdef double a4  =   9.230769230769231e-01  #  12/13
cdef double a5  =   1.000000000000000e+00  #  1
cdef double a6  =   5.000000000000000e-01  #  1/2

cdef double b21 =   2.500000000000000e-01  #  1/4
cdef double b31 =   9.375000000000000e-02  #  3/32
cdef double b32 =   2.812500000000000e-01  #  9/32
cdef double b41 =   8.793809740555303e-01  #  1932/2197
cdef double b42 =  -3.277196176604461e+00  # -7200/2197
cdef double b43 =   3.320892125625853e+00  #  7296/2197
cdef double b51 =   2.032407407407407e+00  #  439/216
cdef double b52 =  -8.000000000000000e+00  # -8
cdef double b53 =   7.173489278752436e+00  #  3680/513
cdef double b54 =  -2.058966861598441e-01  # -845/4104
cdef double b61 =  -2.962962962962963e-01  # -8/27
cdef double b62 =   2.000000000000000e+00  #  2
cdef double b63 =  -1.381676413255361e+00  # -3544/2565
cdef double b64 =   4.529727095516569e-01  #  1859/4104
cdef double b65 =  -2.750000000000000e-01  # -11/40

cdef double r1  =   2.777777777777778e-03  #  1/360
cdef double r3  =  -2.994152046783626e-02  # -128/4275
cdef double r4  =  -2.919989367357789e-02  # -2197/75240
cdef double r5  =   2.000000000000000e-02  #  1/50
cdef double r6  =   3.636363636363636e-02  #  2/55

cdef double c1  =   1.157407407407407e-01  #  25/216
cdef double c3  =   5.489278752436647e-01  #  1408/2565
cdef double c4  =   5.353313840155945e-01  #  2197/4104
cdef double c5  =  -2.000000000000000e-01  # -1/5

cdef class cyfunc:
    cdef double dy[2]

    cdef double* f(self,  double* y):
        return self.dy
    def __cinit__(self):
        pass

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rkf(cyfunc f, np.ndarray[double, ndim=1] times,
          np.ndarray[double, ndim=1] x0,
          double tol=1e-7, double dt_max=-1.0, double dt_min=1e-8):

    # Initialize
    cdef double t = times[0]
    cdef int times_index = 1
    cdef int add = 0
    cdef double end_time = times[len(times) - 1]
    cdef np.ndarray[double, ndim=1] res = np.empty_like(times)
    res[0] = x0[1] # Only storing second variable
    cdef double x[2]
    x[:] = x0

    cdef double k1[2]
    cdef double k2[2]
    cdef double k3[2]
    cdef double k4[2]
    cdef double k5[2]
    cdef double k6[2]
    cdef double r[2]

    while abs(t - times[times_index]) < tol: # if t = 0 multiple times
        res[times_index] = res[0]
        t = times[times_index]
        times_index += 1

    if dt_max == -1.0:
        dt_max = 5. * (times[times_index] - times[0])
    cdef double dt = dt_max/10.0
    cdef double tolh = tol*dt

    while t < end_time:
        # If possible, step to next time to save
        if t + dt >= times[times_index]:
            dt = times[times_index] - t;
            add = 1

        # Calculate Runga Kutta variables
        k1 = f.f(x)
        k1[0] *= dt; k1[1] *= dt;
        r[0] = x[0] + b21 * k1[0]
        r[1] = x[1] + b21 * k1[1]

        k2 = f.f(r)
        k2[0] *= dt; k2[1] *= dt;
        r[0] = x[0] + b31 * k1[0] + b32 * k2[0]
        r[1] = x[1] + b31 * k1[1] + b32 * k2[1]

        k3 = f.f(r)
        k3[0] *= dt; k3[1] *= dt;
        r[0] = x[0] + b41 * k1[0] + b42 * k2[0] + b43 * k3[0]
        r[1] = x[1] + b41 * k1[1] + b42 * k2[1] + b43 * k3[1]

        k4 = f.f(r)
        k4[0] *= dt; k4[1] *= dt;
        r[0] = x[0] + b51 * k1[0] + b52 * k2[0] + b53 * k3[0] + b54 * k4[0]
        r[1] = x[1] + b51 * k1[1] + b52 * k2[1] + b53 * k3[1] + b54 * k4[1]

        k5 = f.f(r)
        k5[0] *= dt; k5[1] *= dt;
        r[0] = x[0] + b61 * k1[0] + b62 * k2[0] + b63 * k3[0] + b64 * k4[0] + b65 * k5[0]
        r[1] = x[1] + b61 * k1[1] + b62 * k2[1] + b63 * k3[1] + b64 * k4[1] + b65 * k5[1]

        k6 = f.f(r)
        k6[0] *= dt; k6[1] *= dt;

        # Find largest error
        r[0] = abs(r1 * k1[0] + r3 * k3[0] + r4 * k4[0] + r5 * k5[0] + r6 * k6[0])
        r[1] = abs(r1 * k1[1] + r3 * k3[1] + r4 * k4[1] + r5 * k5[1] + r6 * k6[1])
        if r[1] > r[0]:
            r[0] = r[1]

        # If error is smaller than tolerance, take step
        tolh = tol*dt
        if r[0] <= tolh:
            t = t + dt
            x[0] = x[0] + c1 * k1[0] + c3 * k3[0] + c4 * k4[0] + c5 * k5[0]
            x[1] = x[1] + c1 * k1[1] + c3 * k3[1] + c4 * k4[1] + c5 * k5[1]
            # Save if at a save time index
            if add:
                while abs(t - times[times_index]) < tol:
                    res[times_index] = x[1]
                    t = times[times_index]
                    times_index += 1
                add = 0

        # Update time stepping
        dt = dt * min(max(0.84 * ( tolh / r[0] )**0.25, 0.1), 4.0)
        if dt > dt_max:
            dt = dt_max
        elif dt < dt_min:  # Equations are too stiff
            return res*0 - 100 # or something

        # ADD STOPPING CONDITION HERE...

    return res

cdef class F(cyfunc):
    cdef double a

    def __init__(self, double a):
        self.a = a

    cdef double* f(self, double y[2]):
        self.dy[0] = self.a*y[1] - y[0]
        self.dy[1] = y[0] - y[1]**2

        return self.dy