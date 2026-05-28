import deepxde as dde
import numpy as np

alpha = dde.Variable(0.5)

geom = dde.geometry.Interval(-1,1)

timedomain = dde.geometry.TimeDomain(0,1)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):

    u_t = dde.grad.jacobian(u, x, i=0, j=1)

    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return u_t - (alpha * u_xx)

x_data = np.random.uniform(-1, 1, (100,1))

t_data = np.random.uniform(0, 1, (100,1))

observe_x = np.hstack((x_data, t_data))

observe_u = np.exp(-0.1 * np.pi**2 * t_data) * np.sin(np.pi * x_data)

ptset = dde.icbc.PointSetBC(observe_x, observe_u, component=0)

data = dde.data.TimePDE(geomtime, pde, [ptset], num_domain=2000, num_boundary=100, num_initial=100)

net = dde.nn.FNN([2] + [32,32,32] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3, external_trainable_variables=[alpha])

variable_cb = dde.callbacks.VariableValue([alpha], period=200)

losshistory, train_state = model.train(iterations=2000, callbacks=[variable_cb])

print("DeepXDE successfully compiled and trained the inverse 1D Diffusion PDE.")