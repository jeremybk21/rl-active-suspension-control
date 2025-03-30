import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import control as ct

from plant_NonlinearQuarterCar import quarter_car_dynamics
from utils import generate_road

# Generate the road profile
t_step = 1e-3
t_sim, z_r = generate_road(t_step, seed=1)

# Nominal System Parameters
m_s = 1200 / 4  # kg
m_u = 50  # kg
k = 15000 / 2  # N/m
b = 1350 / 2
kt = 200000 / 2

# Define the A, B matrices
A = np.array([[0, 1, 0, 0],
              [-k / m_s, -b / m_s, k / m_s, b / m_s],
              [0, 0, 0, 1],
              [k / m_u, b / m_u, -(k + kt) / m_u, -b / m_u]])

B = np.array([0, 1 / m_s, 0, -1 / m_u]).reshape((4, 1))

sys = ct.ss(A, B, np.eye(4), np.zeros((4, 1)))
sys_d = ct.c2d(sys, t_step, method='zoh')
# Define LQR gains
Q = np.zeros((4, 4))
Q[0, 0] = 1000

# Solve Continuous-time Algebraic Riccati Equation (CARE)
R = np.array([[0]])  # Control effort weighting matrix
K, S, E = ct.dlqr(sys_d, Q, R)

# Initialize state variables
x_uncontrolled = np.zeros(4)
x_controlled = np.zeros(4)

# Track variables for plotting
x_track = np.zeros((len(t_sim), len(x_controlled)))
u_track = np.zeros(len(t_sim))

for i in range(len(t_sim)):
    u_lqr = -np.dot(K, x_controlled)
    u_track[i] = u_lqr

    # Integrate the system with odeint
    x_controlled = odeint(quarter_car_dynamics, x_controlled, [0, t_step], args=(z_r[i], u_lqr))[-1]
    x_track[i, :] = x_controlled

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(t_sim, x_track[:, 0], label='Controlled')
plt.ylabel(r'$z_s$')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_sim, x_track[:, 1])
plt.ylabel(r'$\dot{z}_s$')

plt.subplot(4, 1, 3)
plt.plot(t_sim, x_track[:, 2])
plt.ylabel(r'$z_u$')

plt.subplot(4, 1, 4)
plt.plot(t_sim, x_track[:, 3])
plt.ylabel(r'$\dot{z}_u$')
plt.show()

plt.figure()
plt.plot(t_sim, u_track)
plt.ylabel("Control Input u")
plt.show()