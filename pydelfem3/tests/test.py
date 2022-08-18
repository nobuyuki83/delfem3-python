import pydelfem3
import numpy

if __name__ == "__main__":
    s = pydelfem3.sum_as_string(2,3)
    print(s)

    a = numpy.array([0,1,2],dtype=numpy.float64)
    pydelfem3.mult(2., a)
    print(a)
