import deepxde as dde
import numpy as np

print("Running DeepXDE 1D Diffusion")

geom = dde.geometry.Interval(-1,1)

timedomain = dde.geometry.TimeDomain(0,1)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):

    u_t = dde.grad.jacobian(u, x, i=0, j=1)

    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return u_t - (0.1 * u_xx)

def boundary_value(x):

    return np.zeros((len(x), 1))

ic = dde.icbc.IC(geomtime, boundary_value, lambda _, on_initial: on_initial)

bc = dde.icbc.DirichletBC(geomtime, boundary_value, lambda _, on_boundary: on_boundary)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)

net = dde.nn.FNN([2] + [32,32,32] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(iterations=2000)


print("SUCCESS: DeepXDE successfully compiled and trained the 1D Diffusion PDE.")