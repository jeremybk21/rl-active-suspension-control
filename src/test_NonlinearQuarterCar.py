import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import control as ct
from stable_baselines3 import PPO
from stable_baselines3 import TD3

from plant_NonlinearQuarterCar import quarter_car_dynamics
from utils import generate_road, getLQR_K

# Generate the road profile
t_step = 1e-2
t_sim, z_r = generate_road(t_step)

# Import trained model
model = TD3.load("enter model path here")
# Import a second trained model
model2 = PPO.load("enter model path here")

# Initialize state variables
x_uncontrolled = np.zeros(4)
x_controlled = np.zeros(4)
x_controlled2 = np.zeros(4)
x_lqr = np.zeros(4)

# Track variables for plotting
x_track = np.zeros((len(t_sim), len(x_controlled)))
x_track2 = np.zeros((len(t_sim), len(x_controlled2)))
x_track_uncontrolled = np.zeros((len(t_sim), len(x_uncontrolled)))
x_track_lqr = np.zeros((len(t_sim), len(x_lqr)))
u_track = np.zeros(len(t_sim))
u_track2 = np.zeros(len(t_sim))
u_track_lqr = np.zeros(len(t_sim))

# Get LQR K
K = getLQR_K(t_step)

controlled_reward = 0
controlled_reward2 = 0
uncontrolled_reward = 0
lqr_reward = 0

for i in range(len(t_sim)):
    u = model.predict(x_controlled, deterministic=True)[0]
    u_track[i] = u

    # Integrate the system with odeint
    x_controlled = odeint(quarter_car_dynamics, x_controlled, [0, t_step], args=(z_r[i], 1000*u))[-1]
    x_track[i, :] = x_controlled
    controlled_reward += -(x_controlled[0]/0.15)**2 

    # Also simulate the second model
    # u2 = model2.predict(x_controlled2, deterministic=True)[0]
    # u_track2[i] = u2
    # x_controlled2 = odeint(quarter_car_dynamics, x_controlled2, [0, t_step], args=(z_r[i], 1000*u2))[-1]
    # x_track2[i, :] = x_controlled2
    # controlled_reward2 += -(x_controlled2[0]/0.15)**2

    # Also simulate the uncontrolled system
    x_uncontrolled = odeint(quarter_car_dynamics, x_uncontrolled, [0, t_step], args=(z_r[i], 0))[-1]
    x_track_uncontrolled[i, :] = x_uncontrolled
    uncontrolled_reward += -(x_uncontrolled[0]/0.15)**2

    # Also simulate the LQR controlled system
    u_lqr = -np.dot(K, x_lqr)
    u_track_lqr[i] = u_lqr
    x_lqr = odeint(quarter_car_dynamics, x_lqr, [0, t_step], args=(z_r[i], u_lqr))[-1]
    x_track_lqr[i, :] = x_lqr
    lqr_reward += -(x_lqr[0]/0.15)**2

print("Controlled Reward: ", controlled_reward)
print("Controlled2 Reward: ", controlled_reward2)
print("Uncontrolled Reward: ", uncontrolled_reward)
print("LQR Reward: ", lqr_reward)

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(t_sim, x_track[:, 0], label='Controlled')
plt.plot(t_sim, x_track_uncontrolled[:, 0], label='Uncontrolled')
plt.plot(t_sim, x_track_lqr[:, 0], label='LQR')
plt.plot(t_sim, x_track2[:, 0], label='Controlled2')
plt.ylabel(r'$z_s$')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_sim, x_track[:, 1])
plt.plot(t_sim, x_track_uncontrolled[:, 1])
plt.plot(t_sim, x_track_lqr[:, 1])
plt.plot(t_sim, x_track2[:, 1])
plt.ylabel(r'$\dot{z}_s$')

plt.subplot(4, 1, 3)
plt.plot(t_sim, x_track[:, 2])
plt.plot(t_sim, x_track_uncontrolled[:, 2])
plt.plot(t_sim, x_track_lqr[:, 2])
plt.plot(t_sim, x_track2[:, 2])
plt.ylabel(r'$z_u$')

plt.subplot(4, 1, 4)
plt.plot(t_sim, x_track[:, 3])
plt.plot(t_sim, x_track_uncontrolled[:, 3])
plt.plot(t_sim, x_track_lqr[:, 3])
plt.plot(t_sim, x_track2[:, 3])
plt.ylabel(r'$\dot{z}_u$')
plt.show()

plt.figure()
plt.plot(t_sim, 1000*u_track, label='RL')
# plt.plot(t_sim, 1000*u_track2, label='RL2')
plt.plot(t_sim, u_track_lqr, label='LQR')
plt.ylabel("Control Input u")
plt.legend()
plt.show()