from scipy.misc import derivative
from matplotlib import pyplot as plt
import pandas as pd
import math

def f(x):
    return math.sin(x)

def df(x):
    h = 10e-11
    return (f(x + h) - f(x)) / h

def compute_gradient_descent(alpha, eps, epoch,choice):
    x_prev = 0
    y_prev = f(x_prev)
    Y = {x_prev: y_prev}
    for _ in range(epoch):
        if(choice == 0):
            x_new = x_prev - alpha * df(x_prev)
        else:
            x_new = x_prev - alpha * derivative(f,x_prev)
        y_new = f(x_new)
        Y[x_new] = y_new
        if abs(x_new - x_prev) <= eps:
            return Y
        x_prev = x_new
    return Y

x_array = []
y_array = []
for i in range(-50,50):
    x_array.append(i)
    y_array.append(f(i))
plt.plot(x_array,y_array)
plt.title("f(x) = (2x-20)^2")
plt.show()

dy_custom = df(0)
dy_lib = derivative(f, 0)
print(f"df(0) = {dy_custom}")
print(f"derivative(0) = {dy_lib}")

alpha = [0.2, 0.1, 0.001, 0.0001]
eps = [10e-1,10e-2,10e-3,10e-4]

for i in range(0,4):
    for j in range(0,2):
        gradient = compute_gradient_descent(alpha[i], eps[i], 10000,j)
        actual_iteration = len(gradient)
        if(j==0):
            print("\nalpha =",alpha[i],"eps =",eps[i],"derivative function: df(x)")
            plt.title("alpha ="+str(alpha[i])+", eps ="+str(eps[i])+", derivative function: df(x)")
        else:
            print("\nalpha =",alpha[i],"eps =",eps[i],"derivative function: derivative(f,x)")
            plt.title("alpha ="+str(alpha[i])+", eps ="+str(eps[i])+", derivative function: derivative(f,x)")
        print(f"actual_iteration = {actual_iteration}")
        table = pd.DataFrame(gradient.items(), columns=["x", "y"])
        print(table)
        result = table.tail(1)
        print(f"\nGlobal minimum:\n{result}")
        x = list(gradient.keys())
        y = list(gradient.values())
        plt.plot(x_array,y_array)
        plt.plot(x, y, 'ro')
        plt.show()
