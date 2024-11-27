"""
The package is used for compatibility conversion of each related package.
The package only makes relevant compatibility adjustments for the content used in the project,
so that it can receive cupy matrices, convert them into numpy matrices and perform calculations,
and then return cupy matrices.
The package does not use CUDA to implement each related algorithms,
and even a layer of conversion will cause performance problems.
But the performance is generally acceptable,
and a benefit is that the algorithms based on the package can be easily converted from CPU to GPU.
"""
