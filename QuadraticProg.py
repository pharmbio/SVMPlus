# Example: minimize f(x)= x1^2+ 4x2^2 - 8x1 -16x2
#subject to:
# {x_1}+{x_2} <= 5
# {x_1} <= 3
# {x_1} >= 0
# {x_2} >= 0

# rewrite the above in the given standard form:
'''
min 1/2 * 2 [x_1 x_1] [1 0  [x_1  + [-8 -16] [x_1
                       0 4]  x_2]             x_2]

subject to
[1 1 -1 0  [x1     <=   [5 3 0 0 ]
 1 0 0 -1]  x2]
'''
from cvxopt import matrix, solvers
import numpy as np

# QP is of the form
'''
    minimize    (1/2)*x'*P*x + q'*x 
    subject to  G*x <= h      
                A*x = b.
'''
#define the array c
P = 2* matrix([[1.0, 0.],
              [0., 4.]])
q = matrix([-8.0, -16.])

gMatrix = matrix([[-1.0, 1., -1., 0.],
     [1., 0., 0., -1.]])
hMatrix = matrix([5.0, 3., 0., 0.])


sol = solvers.qp(P, q, G= gMatrix, h=hMatrix)
print(sol['x'])


