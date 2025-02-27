import numpy as np

def quarter_car_dynamics(x, t, z_r, u, params= (1200/4, 50, 15000/2, 1000/2, 1500/2, 1200/2, 200000/2)):
    m_s, m_u, k, k_n, b_e, b_c, k_t = params

    x_r = x[2] - x[0]  # Rattle space
    dot_x_r = x[3] - x[1]  # Rattle space velocity

    # Non-linear stiffness and damping forces
    F_s_k = k * x_r + k_n * (x_r)**3
    F_s_b = b_c * dot_x_r if dot_x_r > 0 else b_e * dot_x_r

    # Tire force
    F_t_k = k_t * (z_r - x[2])

    # Sprung and unsprung mass accelerations
    dxdt = np.zeros(4)
    dxdt[0] = x[1]
    dxdt[1] = (1 / m_s) * (F_s_k + F_s_b + u)
    dxdt[2] = x[3]
    dxdt[3] = (1 / m_u) * (-F_s_k - F_s_b - u + F_t_k)

    return dxdt