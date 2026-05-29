import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

geom = dde.geometry.Interval(0,1)

timedomain = dde.geometry.TimeDomain(0,1)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):

    u_t = dde.grad.jacobian(u, x, i=0, j=1)

    u_zz = dde.grad.hessian(u, x, i=0, j=0)

    return u_t - u_zz

def bc_func(x):
    
    return np.zeros((len(x), 1))

bc = dde.icbc.DirichletBC(geomtime, bc_func, lambda _, on_boundary: on_boundary)

def ic_func(x):
    
    return np.ones((len(x), 1))

ic = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)

net = dde.nn.FNN([2] + [32,32,32] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])

losshistory, train_state = model.train(iterations=10000)

z_test = np.linspace(0, 1, 100).reshape(-1, 1)

t_steps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

plt.figure(figsize=(8, 6))

for t_val in t_steps:

    t_test = np.full_like(z_test, t_val)

    X_test = np.hstack((z_test, t_test))
    
    u_pred = model.predict(X_test)
    
    plt.plot(z_test, u_pred, label=f"$T_v$ = {t_val}")

plt.ylim(-0.1, 1.1)
plt.xlabel("Normalized Depth (z/H)")
plt.ylabel("Excess Pore Pressure (u)")
plt.title("Terzaghi Consolidation Isochrones")
plt.legend()
plt.grid(True)

plt.savefig("deepxde_terzaghi_isochrones.png")
plt.show()