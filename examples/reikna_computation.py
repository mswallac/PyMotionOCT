import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.core import Type
from reikna.linalg import MatrixMul
from reikna.transformations import combine_complex
api = cluda.ocl_api()
thr = api.Thread.create()
shape1 = (100, 200)
shape2 = (200, 100)
a_re = numpy.random.randn(*shape1).astype(numpy.float32)
a_im = numpy.random.randn(*shape1).astype(numpy.float32)
b_re = numpy.random.randn(*shape2).astype(numpy.float32)
b_im = numpy.random.randn(*shape2).astype(numpy.float32)
arrays = [thr.to_device(x) for x in [a_re, a_im, b_re, b_im]]
a_re_dev, a_im_dev, b_re_dev, b_im_dev = arrays
a_type = Type(numpy.complex64, shape=shape1)
b_type = Type(numpy.complex64, shape=shape2)
res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.complex64)
dot = MatrixMul(a_type, b_type, out_arr=res_dev)

combine_a = combine_complex(a_type)
combine_b = combine_complex(b_type)

dot.parameter.matrix_a.connect(
combine_a, combine_a.output, a_re=combine_a.real, a_im=combine_a.imag)
dot.parameter.matrix_b.connect(
combine_b, combine_b.output, b_re=combine_b.real, b_im=combine_b.imag)
dotc = dot.compile(thr)
dotc(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

res_reference = numpy.dot(a_re + 1j * a_im, b_re + 1j * b_im)

