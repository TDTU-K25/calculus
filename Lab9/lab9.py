# -*- coding: utf-8 -*-
"""Lab9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13tMr1_EzG-xiu6Pz4WJE0DHwY4Q7l80F

Exercise 1: Calculate the definite integral of functions:
"""

from scipy import integrate
from sympy import *
import math

x = symbols('x', real = True)

f1a = x**3 + 2*x**2 + 3
val_Sf1a = integrate(f1a,(x,1,2))
print(round(val_Sf1a,2))

f1b = 1 / (x**3) + 1 / (x**2) + x * x**(1/2)
val_Sf1b = integrate(f1b,(x,1,4))
print(round(val_Sf1b,2))

f1c = (x**3 + x * x**(1/2) + x) / x**2
val_Sf1c = integrate(f1c,(x,1,4))
print(round(val_Sf1c,2))

f1d = x**3 + (2 / x)
val_Sf1d = integrate(f1d,(x,1,2))
print(round(val_Sf1d,2))

f1e = x**2 * (1/x + 2*x)
val_Sf1e = integrate(f1e,(x,1,2))
print(round(val_Sf1e,2))

f1f = (x**(1/2) - 1) * (x + x**(1/2)+1)
val_Sf1f = integrate(f1f,(x,0,1))
print(round(val_Sf1f,2))

f1g = 1 - 2 / sin(x)**2
val_Sf1g = integrate(f1g,(x,math.pi / 4,math.pi / 2))
print(round(val_Sf1g,2))

f1h = 1 / (sin(x)**2 * cos(x)**2)
val_Sf1h = integrate(f1h,(x,math.pi / 6,math.pi / 4))
print(round(val_Sf1h,2))

f1i = math.e**x * (1 - (math.e**(-x) / cos(x)**2))
val_Sf1i = integrate(f1i,(x,0,math.pi / 4))
print(round(val_Sf1i,2))

f1j = math.e**x * (2 + (math.e**(-x) / math.e**x))
val_Sf1j = integrate(f1j,(x,0,math.log(2)))
print(round(val_Sf1j,2))

f1k = 2**x + 2 / x
val_Sf1k = integrate(f1k,(x,1,2))
print(round(val_Sf1k,2))

f1l = x**2 * (x-1)**2
val_Sf1l = integrate(f1l,(x,0,1))
print(round(val_Sf1l,2))

f1m = 1 / (x * (x+1))
val_Sf1m = integrate(f1m,(x,1,2))
print(round(val_Sf1m,2))

f1n = abs(1-x)
val_Sf1n = integrate(f1n,(x,0,2))
print(round(val_Sf1n,2))

f1o = abs(2*x - x**2)
val_Sf1o = integrate(f1o,(x,1,4))
print(round(val_Sf1o,2))

f1p = (x**2 - 3*x + 2)**(1/2)
val_Sf1p = integrate(f1p,(x,2,4))
print(round(val_Sf1p,2))

# Error
f1q = (1 + cos(2*x)) ** (1/2)
#val_Sf1q = integrate(f1q,(x,0,math.pi))
#print(round(val_Sf1q,2))

"""Exercise 2: Calculate the definite integrals and plot graphs of functions:"""

from scipy import integrate
from sympy import *
import numpy as np

x = symbols('x')
y = symbols('y')

# Cách 1
f2d = x**2 * y
val_Sf2d = integrate(f2d,(x,0,3),(y,1,2))
print(val_Sf2d)

"""Exercise 3"""

from scipy import integrate
from sympy import *
import math

x = symbols('x', real = True)

f3a = x**2 - 1
val_Sf3a = integrate(f3a,(x,0,math.sqrt(3)))
res_f3a = val_Sf3a / (math.sqrt(3) - 0)
print(res_f3a)

f3b = -x**2 / 2 
val_Sf3b = integrate(f3b,(x,0,3))
res_f3b = val_Sf3b / (3 - 0)
print(res_f3b)

f3c = -3*x**2 - 1
val_Sf3c = integrate(f3c,(x,0,1))
res_f3c = val_Sf3c / (1 - 0)
print(res_f3c)

f3d = x**2 - x
val_Sf3d = integrate(f3d,(x,-2,1))
res_f3d = val_Sf3d / (1 - (-2))
print(res_f3d)

"""Exercise 4"""

f4a = cos(x)*x**2
f4b = math.e**(-x**2/2)
f4c = sin(x)

val_Sf4a = integrate(f4a,(x,-4,9))
val_Sf4b = integrate(f4b,(x,-oo,+oo))
val_Sf4c = integrate(f4c,(x,-oo,+oo))

print(val_Sf4a)
print(val_Sf4b)
print(val_Sf4c)

"""Exercise 5"""

f5 = 160 - 32*x
val_Sf5 = integrate(f5, (x,0,8))
print(val_Sf5)