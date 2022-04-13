# -*- coding: utf-8 -*-
"""Lab5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18WaNd4IPcLwOsAaXEbQZQtvmoSBG8zpV

Exercise 9: Find f'(x) and use it to find equations of the tangent lines to curve f(x) = 4*x**2 − x**3 at points (2, 8) and (3, 9). Illustrate your result by graphing the curve and the tangent lines on the same graph.
"""

import sympy as sp

x = sp.symbols('x')

f = 4 * x**2 - x**3
df = sp.diff(f,x)

slope_1 = df.subs(x,2)
slope_2 = df.subs(x,3)

y_tangentline_1 = slope_1 * (x - 2) + 8
y_tangentline_2 = slope_2 * (x - 3) + 9

sp.plot(f, y_tangentline_1, y_tangentline_2, (x, -3, 3))

"""Exercise 10"""

import sympy as sp

x = sp.symbols('x')
f = (x-1)**(1/3)
df = sp.diff(f,x)

print("10a - Derivative of f(x) at x = 1: ",df.subs(x,1)) # zoo: vô cùng trong không gian số phức -> không tồn tại giới hạn tại x = 1

dx = sp.symbols('dx')
x0 = -2
f = -(x+2)
df = ( f.subs(x, x0 + dx) - f.subs(x, x0) ) / dx
lml = sp.limit(df, dx, 0, '-')

f = x+2
df = ( f.subs(x, x0 + dx) - f.subs(x, x0) ) / dx
lmr = sp.limit(df, dx, 0, '+')

if lmr == lml:
  print("10b - Derivative of f(x) at x = -2: ",lmr)
else:
  print("10b - f(x) không khả vi tại x = -2")

x0 = 0
f = x**2
df = ( f.subs(x, x0 + dx) - f.subs(x, x0) ) / dx
lmr = sp.limit(df, dx, 0, '+')

if lmr == 0:
  print("10c - Derivative of f(x) at x = 0: ",lmr)
else:
  print("10c - f(x) không khả vi tại x = 0")

"""Exercise 11: Determine whether f'(0) exists (f(x) is differentiable at x = 0)"""

import sympy as sp

x = sp.symbols('x')
f = x * sp.sin(1/x)

x0 = 0
dx = sp.symbols('dx')
df0 = ( f.subs(x, x0 + dx) - 0 ) / dx
print("11a - Derivative of f(x) at x = 0: ",sp.limit(df0, dx, 0))
# f'(0) không tồn tại => f(x) không khả vi tại x = 0

f = x**2 * sp.sin(1/x)

df0 = ( f.subs(x, x0 + dx) - 0 ) / dx
print("11b - Derivative of f(x) at x = 0: ",sp.limit(df0, dx, 0))
# f'(0) tồn tại => f(x) khả vi tại x = 0

"""Exercise 12 Suppose that it costsc(x) = x**3 − 6*x**2 + 15*x dollars to produce x radiators when 8 to 30 radiatord are produced. Your shop currently produces 10
radiators a day. Write a program to compute how much extra will it cost to produce one more radiator
a day.
"""

from sympy import *
import numpy as np
x = symbols('x')
f12 = x**3 - 6*x**2 + 15*x
df12 = f12.diff(x)
df12_x = lambdify((x),df12)
print("",df12_x(10))

"""Exercise 13 Suppose that the revenue from selling x washing machines is r(x) = 20000 * (1 − 1/x) dollars
Write a program to find the marginal revenue when 100 machines are produced.
"""

from sympy import *
import numpy as np
x = symbols('x')
print("Exercise 13")
f13 = 20000*(1-1/x)
df13 = f13.diff(x)
df13_x = lambdify((x),df13)
print("Doanh thu cận biên khi SX 100 máy giặt: ",df13_x(100))

"""Exercise 14 When a bactericide was added to a nutrient broth in which bacteria were growing, the bacterium population continued to grow for a while, but then stopped growing and began to decline.
The size of the population at time t (hours) was
b = 10**6 + 10**4*t − 10**3*t**2
Find the growth rates at
(a) t = 0 hours
(b) t = 5 hours
(c) t = 10 hours
"""

from sympy import *
import numpy as np
print("Exercise 14")
t = Symbol('t')
f14 = 10**6 + 10**4*t - 10**3*t**2
df14 = f14.diff(t)
df14_t = lambdify((t),df14)
print("b'(t) = ",df14)
print("t = 0, b'(t) = ",df14_t(0))
print("t = 5, b'(t) = ",df14_t(5))
print("t = 10, b'(t) = ",df14_t(10))

"""Exercise 15 A rock thrown vertically upward from the surface of the moon at a velocity of 24 m/sec reaches a height of s = 24*t − 0.8*t**2 m in t sec. Write a program to
(a) Find the rock’s velocity and acceleration at time t.
(b) How long does it take the rock to reach its highest point?
(c) How high does the rock go?
"""

from sympy import *
import numpy as np
print("Exercise 15")
f15 = 24*t - 0.8*t**2
df15 = f15.diff(t)
print("df15=",df15)

"""Exercise 16 Write a program to implement Newton algorithm, find the approximation of the root func-tion. Perform Newton-Raphson method by
(a) f(x) = 2*x**3 + 3*x − 1 with starting interval p0 = 2 and a tolerance e = 10**−8. Then, put the results in a table and plot the graph.
(b) f(x) = x**3 − 4, perform 3 iterations with starting point p0 = 2. Then, put the results in a table and plot the graph.
"""

from sympy import *
import numpy as np
print("Exercise 16")
x = Symbol('x')
f16a = 2*x**3 + 3*x - 1
g = f16a.diff(x)
print("f(x)=2*x**3 + 3*x - 1, df(x)=",g)