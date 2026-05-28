import deepxde as dde
import numpy as np

geom = dde.geometry.Interval(-1,1)

timedomain = dde.geometry.TimeDomain(0,1)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):

    u_t = dde.grad.jacobian(u, x, i=0, j=1)

    u_x = dde.grad.jacobian(u, x, i=0, j=0)

    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return u_t + u * u_x - (0.01/np.pi) * u_xx

def ic_func(x):

    return -np.sin(np.pi * x[:, 0:1])

def bc_func(x):

    return np.zeros((len(x), 1))

ic = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

bc = dde.icbc.DirichletBC(geomtime, bc_func, lambda _, on_boundary: on_boundary)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540, num_boundary=100, num_initial=100)

net = dde.nn.FNN([2] + [50, 50, 50, 50] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(iterations=2000)

print("DeepXDE successfully compiled and trained on the Burgers Equation")
