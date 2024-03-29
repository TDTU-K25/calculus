# -*- coding: utf-8 -*-
"""52100943_Lab6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12bf7pFDe0MqqUp6fjmARYDHRXIC5HrGl

Exercise 3: Find the Taylor series expension of these function
"""

from sympy import *
import math

x = symbols('x')

f3a = cos(x)
f3b = ln(x)
f3c = math.e**x

taylor_poly_a = f3a.series(x, pi/3, 6)
taylor_poly_b = f3b.series(x, 2, 10)
taylor_poly_c = f3c.series(x, 3, 12)

print("a) Taylor polynomial {}".format(taylor_poly_a))
print("b) Taylor polynomial {}".format(taylor_poly_b))
print("c) Taylor polynomial {}".format(taylor_poly_c))

"""Exercise 4: Find the Maclaurin series expansion of these function"""

from sympy import * 
import math


f4a = cos(x)
f4b = math.e**x
f4c = 1 / (1 - x)
f4d = 1 / tan(x)

df4a = diff(f4a,x)
maclaurin_poly_a = df4a.series(x, 0, 6)
print("a) Maclaurin polynomial {}".format(maclaurin_poly_a))

df4b = diff(f4b,x)
maclaurin_poly_b = df4b.series(x, 0, 12)
print("b) Maclaurin polynomial {}".format(maclaurin_poly_b))

df4c = diff(f4c,x)
maclaurin_poly_c = df4c.series(x, 0, 12)
print("c) Maclaurin polynomial {}".format(maclaurin_poly_c))

df4d = diff(f4d,x)
maclaurin_poly_d = df4d.series(x, 0, 12)
print("d) Maclaurin polynomial {}".format(maclaurin_poly_d))

"""Exercise 5: Find the limit of the following sequences"""

from sympy import *
import math

n = symbols('n')

f5a = (4 * n**2 + 1) / (3 * n**2 + 2)
f5b = (sqrt(n**2 + 1) - n)
f5c = sqrt(2*n + sqrt(n)) - sqrt(2*n + 1)
f5d = (3 * 5**n - 2**n) / (4**n + 2 * 5**n)
f5e = ( n * sin(sqrt(n)) ) / (n**2 + n - 1)

lim_f5a = limit(f5a, n, math.inf)
lim_f5b = limit(f5b, n, math.inf)
lim_f5c = limit(f5c, n, math.inf)
lim_f5d = limit(f5d, n, math.inf)
lim_f5e = limit(f5e, n, math.inf)

print("a) The limit of " + str(f5a) + " is",lim_f5a)
print("b) The limit of " + str(f5b) + " is",lim_f5b)
print("c) The limit of " + str(f5c) + " is",lim_f5c)
print("d) The limit of " + str(f5d) + " is",lim_f5d)
print("e) The limit of " + str(f5e) + " is",lim_f5e)

"""Exercise 6: Determine whether the sequence converges or diverges"""

from sympy import *
import math

n = symbols('n', positive=True)

def converges_diverges(f):
  lim_f = limit(f, n, oo)
  if(lim_f == oo):
    print("Dãy {} phân kỳ".format(f))
  else:
    print("Dãy {} hội tụ về giá trị: {}".format(f,lim_f))
  
f6a = 1 - (0.2)**n
converges_diverges(f6a)

f6b = n**3 / (n**3 + 1)
converges_diverges(f6b)

f6c = (3 + 5*n**2) / (n + n**2)
converges_diverges(f6c)

f6d = n**3 / (n + 1)
converges_diverges(f6d)

f6e = math.e**(1/n)
converges_diverges(f6e)

f6f = sqrt((n+1) / (9*n + 1))
converges_diverges(f6f)

f6g = ((-1)**(n+1) * n) / (n + sqrt(n))
#converges_diverges(f6g) # Error

f6h = tan((2*n*math.pi) / (1 + 8*n))
converges_diverges(f6h)

f6i = factorial(2*n - 1) / factorial(2*n + 1)
converges_diverges(f6i)

f6j = log(2*n**2 + 1) - log(n**2 + 1)
converges_diverges(f6j)

"""Exercise 7: Find the first five terms of the sequence following:"""

from sympy import *
from matplotlib import pyplot as plt
import numpy as np
import math

def an(f,k):
  f_n = lambdify(n,f)
  for x in range(0,k):
    print("n = " + str(x) + " => an = " + str(f_n(x)))

def draw_graph(f,value,title):
  f_n = lambdify(n,f)(value)
  plt.plot(value,f_n)
  plt.title(title)
  plt.show()

k = 5

f7a = 1 - 0.2**n
print("7a) a(n) = " + str(f7a) + ", n = 0,1,...," + str(k-1))
an(f7a,k)

print()

f7b = 2*n/(n**2 + 1)
print("7b) a(n) = " + str(f7b) + ", n = 0,1,...," + str(k-1))
an(f7b,k)

print()

f7c = (-1)**(n-1)/(5**n)
print("7c) a(n) = " + str(f7c) + ", n = 0,1,...," + str(k-1))
an(f7c,k)

print()

f7d = 1/factorial(n + 1)
print("7d) a(n) = " + str(f7d) + ", n = 0,1,...," + str(k-1))
an(f7d,k)

print()

print("7e) a(n + 1) = 5 * a(n) - 3" + ", n = 1,2,...," + str(k))
f7e_a_n = 1 # a(1) = 1
for x in range(1, k + 1):
  f7e_a_n_plus_1 = 5 * f7e_a_n - 3
  f7e_a_n = f7e_a_n_plus_1
  print("n = " + str(x) + " => a(n+1) = " + str(f7e_a_n_plus_1))

print()

print("7f) a(n + 1) = a(n) / a(n) + 1" + ", n = 1,2,...," + str(k))
f7f_a_n = 2 # a(1) = 2
for x in range(1, k + 1):
  f7f_a_n_plus_1 = f7f_a_n / (f7f_a_n + 1)
  f7f_a_n = f7f_a_n_plus_1
  print("n = " + str(x) + " => a(n+1) = " + str(f7f_a_n_plus_1))

print()

value = np.arange(0,5,0.1)
title = 'Câu a'
draw_graph(f7a,value,title)

print()

value = np.arange(0,100,1)
title = 'Câu b'
draw_graph(f7b,value,title)

print()

# Error
"""
value = np.arange()
title = 'Câu c'
draw_graph(f7c,value,title)
"""

print()

value = np.arange(0,10,0.1)
title = 'Câu d'
draw_graph(f7d,value,title)

print()

"""Exercise 8: Using a graph of the sequence to determine whether the sequence is convergent or divergent."""

from sympy import *
from matplotlib import pyplot as plt
import numpy as np
import math

n = symbols('n')

def plot(a,k):
  value = np.arange(1,k+1,1)

  a_n = lambdify(n, a)(value)

  plt.plot(value,a_n)

  plt.show()

  print()

f8a = 1 - (-2 / math.e)**n
plot(f8a,25)

f8b = sqrt(n) * sin(math.pi / sqrt(n))
plot(f8b,10000)

f8c = sqrt((3+2*n**2) / (8*n**2 + n))
plot(f8c,100)

f8d = (n**2 * cos(n)) / (1+n**2)
plot(f8d,100)

# Eroor
"""
f8e = (factorial(2*n) / (2**n * factorial(n))) / factorial(n)
plot(f8e,10)
"""

# Eroor
"""
f8f = (factorial(2*n) / (2**n * factorial(n))) / (2*n)**n
plot(f8f,10)
"""

"""Exercise 9: Determinate if the following series is convergent or divergent."""

from sympy import *
import math

n = symbols('n')

def convergent_divergent(series):
  sqrt_n_series = series**(1/n)
  if(limit(sqrt_n_series,n,oo) > 1):
    print("Chuỗi {} phân kì".format(series))
  elif(limit(sqrt_n_series,n,oo) < 1):
    print("Chuỗi {} hội tụ".format(series))
  else:
    print("Chưa kết luận được")

f9a = 4**n
convergent_divergent(f9a)

f9b = 5 / (2**n)
convergent_divergent(f9b)