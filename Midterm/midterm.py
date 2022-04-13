from sympy import * ## KHONG XOA
import numpy as np ## KHONG XOA 

global x, y, z, t ## KHONG XOA
x, y, z, t = symbols("x, y, z, t") ## KHONG XOA     

def req1(f, g, a):  ## KHONG XOA
    g_x = lambdify(x, g)

    df = diff(f, x, 1) 
    dg = diff(g, x, 1) 

    f1 = df + dg 
    try:
        if f1.subs(x,a) == zoo or f1.subs(x,a) == nan:
            f1 = None
        elif(np.iscomplex(f1.subs(x,a))):
            f1 = None
        else:
            f1 = float(f1.subs(x,a))
            f1 = round(f1,2) 
    except ZeroDivisionError:
        f1 = None 

    f2 = df * g + dg * f 
    try:
        if f2.subs(x,a) == zoo or f2.subs(x,a) == nan:
            f2 = None
        elif(np.iscomplex(f2.subs(x,a))):
            f2 = None
        else:
            f2 = float(f2.subs(x,a))
            f2 = round(f2,2)
    except ZeroDivisionError:
        f2 = None

    g_a = g_x(a)
    df_g = df.subs(x,g_a) 
    dg_a = dg.subs(x,a) 
    f3 = df_g * dg_a
    try:
        if f3.subs(x,a) == zoo or f3.subs(x,a) == nan:
            f3 = None
        elif(np.iscomplex(f3.subs(x,a))):
            f3 = None
        else:
            f3 = float(f3.subs(x,a))
            f3 = round(f3,2) 
    except ZeroDivisionError:
        f3 = None 

    f4 = (df * g - dg * f) / g**2
    try:
        if f4.subs(x,a) == zoo or f4.subs(x,a) == nan:
            f4 = None
        elif(np.iscomplex(f4.subs(x,a))):
            f4 = None
        else:
            f4 = float(f4.subs(x,a))
            f4 = round(f4,2)
    except ZeroDivisionError:
        f4 = None 

    return f1,f2,f3,f4

def req2(f, a, b, c):  ## KHONG XOA
    df_x = diff(f,x,1) 
    df_y = diff(f,y,1) 
    df_z = diff(f,z,1)

    try:
        if f.subs(x,a).subs(y,b).subs(z,c) == nan or f.subs(x,a).subs(y,b).subs(z,c) == zoo or np.iscomplex(f.subs(x,a).subs(y,b).subs(z,c)):
            return None
        else:
            f_P = float(f.subs(x,a).subs(y,b).subs(z,c))
    except ZeroDivisionError:
        return None

    try:
        if df_x.subs(x,a).subs(y,b).subs(z,c) == nan or df_x.subs(x,a).subs(y,b).subs(z,c) == zoo or np.iscomplex(df_x.subs(x,a).subs(y,b).subs(z,c)):
            return None
        else:
            df_x_P = float(df_x.subs(x,a).subs(y,b).subs(z,c))
    except ZeroDivisionError:
        return None

    try:
        if df_y.subs(x,a).subs(y,b).subs(z,c) == nan or df_y.subs(x,a).subs(y,b).subs(z,c) == zoo or np.iscomplex(df_y.subs(x,a).subs(y,b).subs(z,c)):
            return None
        else:
            df_y_P = float(df_y.subs(x,a).subs(y,b).subs(z,c))
    except ZeroDivisionError:
        return None

    try:
        if df_z.subs(x,a).subs(y,b).subs(z,c) == nan or df_z.subs(x,a).subs(y,b).subs(z,c) == zoo or np.iscomplex(df_z.subs(x,a).subs(y,b).subs(z,c)):
            return None
        else:
            df_z_P = float(df_z.subs(x,a).subs(y,b).subs(z,c))
    except ZeroDivisionError:
        return None

    tangent_f = df_x_P * (x - a) + df_y_P * (y - b) + df_z_P * (z - c) + f_P  

    return tangent_f

def req3(w, f1, f2, f3, a):  ## KHONG XOA
    w_t = w.subs(x,f1).subs(y,f2).subs(z,f3) 
    dw = diff(w_t,t,1) 
    dw_a = dw.subs(t,a) 
    try:
        if dw_a == nan or dw_a == zoo:
            dw_a = None
        elif(np.iscomplex(dw_a)):
            dw_a = None
        else:
            dw_a = float(dw_a)
    except ZeroDivisionError:
        dw_a = None
    return dw_a

def req4(a, b, n):  ## KHONG XOA
    def Combination(i, n):
        C = float(factorial(n) / (factorial(n - i) * factorial(i)))
        return C

    sum = 0.0
    i = 0

    while(i <= n):
        sum += Combination(i,n) * a**(n-i) * b**i
        i = i + 1
    return sum

def req5(f):  ## KHONG XOA
    df_x = diff(f,x,1) 
    df_y = diff(f,y,1) 
    M = solve((df_x, df_y), dict = True) 

    df_x_2 = diff(df_x,x,1) 
    df_y_2 = diff(df_y,y,1) 
    df_x_y = diff(df_x,y,1)

    localMaxima = [] 
    localMinima = [] 
    saddlePoint = [] 

    for M0 in M:
        if(M0[x].is_real and M0[y].is_real):
            A = lambdify((x,y), df_x_2)(M0[x],M0[y])
            B = lambdify((x,y), df_x_y)(M0[x],M0[y])
            C = lambdify((x,y), df_y_2)(M0[x],M0[y])
            delta = A*C - B**2 
            if(delta > 0 and A > 0):
                localMinima.append((M0[x],M0[y]))
            elif(delta > 0 and A < 0):
                localMaxima.append((M0[x],M0[y]))
            elif(delta < 0):
                saddlePoint.append((M0[x],M0[y]))

    return localMinima , localMaxima, saddlePoint

def req6(message, x, y, z):  ## KHONG XOA
    Plain_text = message

    f = abs(x**2 - y**2 - z)
    Secret_key = format(f, "08b") 

    Cirpher_text_char_list = [] 

    for char in Plain_text:
        ascii_char = ord(char) 
        binary_char = format(ascii_char, "08b") 
        Cirpher_text_char = ''.join([str(int(let1) ^ int(let2)) for let1, let2 in zip(Secret_key, binary_char)]) 
        Cirpher_text_char = chr(int(Cirpher_text_char, 2))
        Cirpher_text_char_list.append(Cirpher_text_char) 
    
    Cirpher_text = ''.join([str(char) for char in Cirpher_text_char_list])

    return Cirpher_text

def req7(xp, yp, xc):  ## KHONG XOA
    sum_x = 0
    for x in xp:
        sum_x += x

    sum_xMultiplex = 0
    for x in xp:
        sum_xMultiplex += x**2

    sum_xMultipley = 0
    for x, y in zip(xp, yp):
        sum_xMultipley += (x * y)

    sum_y = 0
    for y in yp:
        sum_y += y

    n = len(xp) 

    m = ((sum_x * sum_y) - (n * sum_xMultipley)) / (sum_x**2 - (n  * sum_xMultiplex))

    b = (1 / n) * (sum_y - m * sum_x)

    y_xc = float(m * xc + b)
    y_xc = round(y_xc,2)

    return y_xc

def req8(f, eta, xi, tol): ## KHONG XOA
    df = diff(f,x,1)
    df_x = lambdify(x, df)
    x_t = xi
    while 1:
        try:
            x_t_plus_1 = x_t - eta*df_x(x_t)
        except ZeroDivisionError:
            return None
        x_t = x_t_plus_1
        if(abs(df_x(x_t_plus_1)) < tol):
            if(x_t == nan or x_t == zoo or np.iscomplex(x_t)):
                return None
            x_t = float(x_t)
            x_t = round(x_t,2)
            break
    return x_t

