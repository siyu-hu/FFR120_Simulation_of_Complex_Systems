import numpy as np
import matplotlib.pyplot as plt

def compute_msd(positions):
    """
    Compute the mean square displacement of the disk 

    Parameters
    - positions (list) : disk position

    Return
    - msd (list)
    - time_steps (numpy.ndarray)
    """
    N = len(positions)
    msd = []
    time_steps = np.arange(1, N)
    for n in time_steps:
        displacements = []
        for i in range(N - n):
            dx = positions[i + n][0] - positions[i][0]
            dy = positions[i + n][1] - positions[i][1]
            displacement = dx ** 2 + dy ** 2
            displacements.append(displacement)
        msd_value = np.mean(displacements)
        msd.append(msd_value)
    return msd, time_steps


def plot_msd(msd_values, time_values, dt):
    """
    Plot MSD-timevalus image

    Parameter
    - msd_values (list): MSD list 
    - time_values (numpy.ndarray): timesteps value
    - dt (float): timestep
    """
    plt.figure()
    time = time_values * dt
    plt.plot(time, msd_values, label='MSD')
    plt.xlabel('Time')
    plt.ylabel('MSD')
    plt.title('Mean Square Displacement')
    plt.legend()
    plt.grid(True)
    plt.savefig('msd_plot.png')


def plot_trajectory(positions):
    """
    Plot the trajectory of the disk

    Parameter
    positions (list): disk's positon
    """
    positions = np.array(positions)
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], label='Disk Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Disk Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_plot.png')
