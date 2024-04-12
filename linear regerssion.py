import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#Get the training data

dfx = pd.read_csv('./csv/linearX.csv')
dfy = pd.read_csv('./csv/linearY.csv')

x = dfx.values
y = dfy.values

x = x.reshape((-1,)) #covert to numpy
y = y.reshape((-1,)) #covert to numpy
print(x.shape)
print(y.shape)

# Visualise and Preprocess
plt.scatter(x,y)
X = (x-x.mean())/x.std()#mean and standard derivation
Y = y
plt.scatter(X,Y)
plt.show()

#Gradient Descent Algorithm
def hypothesis(x, theta):
    return theta[0] + theta[1] * x #theta[0] = intersect(c) theta[1] = slop(m)


def error(X, Y, theta):#error fun
    m = X.shape[0]
    error = 0

    for i in range(m):
        hx = hypothesis(X[i], theta)
        error += (hx - Y[i]) ** 2

    return error


def gradient(X, Y, theta):#gradient
    grad = np.zeros((2,))
    m = X.shape[0]

    for i in range(m):
        hx = hypothesis(X[i], theta)
        grad[0] += (hx - Y[i])
        grad[1] += (hx - Y[i]) * X[i]

    return grad


# Algorithm
def gradientDescent(X, Y, learning_rate=0.001):
    theta = np.array([-2.0, 0.0])

    itr = 0
    max_itr = 100

    error_list = []
    theta_list = []

    while (itr <= max_itr):
        grad = gradient(X, Y, theta)
        e = error(X, Y, theta)
        error_list.append(e)

        theta_list.append((theta[0], theta[1]))
        theta[0] = theta[0] - learning_rate * grad[0]
        theta[1] = theta[1] - learning_rate * grad[1]

        itr += 1

    return theta, error_list, theta_list

final_theta, error_list,theta_list = gradientDescent(X,Y)

plt.plot(error_list)
plt.show()

print(final_theta)

### Plot the line for testing data

xtest = np.linspace(-2,6,10)
print(xtest)

plt.scatter(X,Y,label='Training Data')
plt.plot(xtest,hypothesis(xtest,final_theta),color='orange',label="Prediction")
plt.legend()
plt.show()




#Visualising Gradient Descent

# 3D Loss Plot


# ax = fig.add_subplot(111,project='3d')
T0 = np.arange(-2, 3, 0.01)
T1 = np.arange(-2, 3, 0.01)
T0, T1 = np.meshgrid(T0, T1)
J = np.zeros(T0.shape)
m = T0.shape[0]
n = T0.shape[1]

for i in range(m):
    for j in range(n):
        J[i, j] = np.sum((Y - T1[i, j] * X - T0[i, j]) ** 2)

fig = plt.figure()
axes = fig.gca(projection='3d')

theta_list = np.array(theta_list)
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list, c='k')

axes.plot_surface(T0, T1, J, cmap='rainbow', alpha=.5)
plt.show()

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.contour(T0,T1,J,cmap='rainbow')
axes.set_xlim([-2,2])
axes.set_ylim([-2,2])
axes.scatter(theta_list[:,0],theta_list[:,1],error_list,c='k',marker='^')
plt.title("3D Contour")
plt.show()

plt.contour(T0,T1,J)
plt.title("2D Contour")
th = np.array(theta_list)
plt.scatter(th[:,0],th[:,1],marker='>',label='Trajectory')
plt.legend()
plt.show()