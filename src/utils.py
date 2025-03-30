import numpy as np
import control as ct

def generate_road(t_step, s_max=200, seed=1):
    np.random.seed(seed)

    # User inputs
    w = 2  # Waviness
    Om_0 = 1  # Reference wavenumber
    Om_1 = 2 * np.pi / 100  # Minimum sine wave frequency range
    Om_N = 2 * np.pi / 0.1  # Maximum sine wave frequency range
    psd_0 = 16e-6  # Power spectral density (country road)
    N = 20000  # Number of sine waves
    speed = 60  # km/h
    speed_ms = speed / 3.6  # m/s

    # Calculations
    ds = s_max / N  # Equal path interval
    s_now = np.arange(0, s_max, ds)
    t = s_now / speed_ms
    dOm = (Om_N - Om_1) / N  # Equal frequency interval
    Om = np.arange(Om_1, Om_N, dOm)
    psi = 2 * np.pi * np.random.rand(len(Om))  # Random phase shifts
    psd = psd_0 * (Om / Om_0) ** -w  # Power spectral density
    Ar = np.sqrt(2 * psd * dOm)  # Amplitude

    # Generate the trajectory
    Z_R = np.zeros(len(t))
    for i in range(len(t)):
        Z_R[i] = np.sum(Ar * np.sin(Om * t[i] - psi))

    # Time scale of the simulation
    t_step = 1e-3
    t_end = t[-1]
    t_sim = np.arange(0, t_end, t_step)

    z_r = np.interp(t_sim, t, Z_R)
    return t_sim, z_r

def getLQR_K(t_step):
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
    return K