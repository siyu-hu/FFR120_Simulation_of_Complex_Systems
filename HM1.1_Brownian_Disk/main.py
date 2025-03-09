import numpy as np
from scipy.optimize import curve_fit
from simulation import (
    initialize_particles,
    initialize_disk,
    remove_overlapping_particles,
    compute_force, 
)
from compute_and_plot import compute_msd, plot_trajectory, plot_msd
import matplotlib.pyplot as plt

print("Starting program")

# Parameters of particles
m = 1.0
v0 = 10.0
N_part = 625

# Parameters of disk
m_disk = 10.0
R_disk = 10.0

# Box size
L = 260.0

# Lennard-Jones potential parameters
sigma = 1.0
epsilon = 1.0

# Simulation parameters
T_tot = 100.0
dt = 0.005
num_steps = int(T_tot / dt)
cutoff_radius = 2 * R_disk  # neighbourhood

# Initialize particles and disk
particles = initialize_particles(N_part, L, v0, m)
disk = initialize_disk(L, m_disk, R_disk)
particles = remove_overlapping_particles(particles, disk, sigma)

disk_positions = []


# Apply simulation

#DEBUG
plt.ion() 
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Disk Trajectory')
trajectory_plot, = ax.plot([], [], 'b-')  
disk_point_plot, = ax.plot([], [], 'go', markersize=20)
neighborhood_plot, = ax.plot([], [], 'ro', markersize=2, linestyle='None') 
particles_plot, = ax.plot([], [], 'bo', markersize=1, linestyle='None') 

for step in range(num_steps):
    total_force = np.array([0.0, 0.0])
    neighborhood_particles = [] 

    for particle in particles:
        distance = np.linalg.norm(particle.position - disk.position)
        if distance <= cutoff_radius:
            neighborhood_particles.append(particle)
    

    for particle in neighborhood_particles:

        # compute force (particles to disk)  and update paricle velocity and position
        half_position_part = particle.position + 0.5 * particle.velocity * dt
        force = compute_force(particle, disk, epsilon, sigma) # has same director as [ disk ->> particle ]
        next_velocity_part = particle.velocity + (force / m) * dt
        next_position_part = half_position_part + 0.5 * next_velocity_part * dt

        particle.position = next_position_part
        particle.velocity = next_velocity_part
        
        particle.apply_boundary_conditions(L)
        total_force += force   # total force of  [particle ->> disk] 
    
    # update disk velocity and position
    force_on_disk = -total_force
    half_position_disk = disk.position + 0.5 * dt * disk.velocity
    next_velocity_disk = disk.velocity + (force_on_disk / m_disk) * dt
    next_position_disk = half_position_disk + 0.5 * dt *  next_velocity_disk 
    
    disk.position = next_position_disk
    disk.apply_boundary_conditions(L)
    disk.velocity = next_velocity_disk
    disk_positions.append(disk.position.copy())

    # DEBUG 
    print(f"Num of time steps = {step + 1}, Total force = {total_force}, Disk Position = {disk_positions[-1]}")

    x_positions = [pos[0] for pos in disk_positions]
    y_positions = [pos[1] for pos in disk_positions]
    trajectory_plot.set_data(x_positions, y_positions)
    disk_point_plot.set_data([disk.position[0]], [disk.position[1]])
   
    neighborhood_x = [p.position[0] for p in neighborhood_particles]
    neighborhood_y = [p.position[1] for p in neighborhood_particles]
    neighborhood_plot.set_data(neighborhood_x, neighborhood_y)

    other_particles = [p for p in particles if p not in neighborhood_particles]
    particles_x = [p.position[0] for p in other_particles]
    particles_y = [p.position[1] for p in other_particles]
    particles_plot.set_data(particles_x, particles_y)

    plt.draw()
    plt.pause(0.1)

    # DEBUG: Plot every 10 steps
    # if (step + 1) % 20 == 0:
       
    #     plt.figure()
    #     plt.scatter(disk.position[0], disk.position[1], color='red', s=100)
    #     particle_positions = np.array([particle.position for particle in particles])
    #     plt.scatter(particle_positions[:, 0], particle_positions[:, 1], color='blue', s=10)
    #     plt.xlim(0, L)
    #     plt.ylim(0, L)
    #     plt.xlabel('X Position')
    #     plt.ylabel('Y Position')
    #     plt.title(f'Positions of Disk and Particles at Step {step + 1}')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show(block=False)
    #     plt.pause(2)  # Pause for 1 second to show the plot
    #     plt.close()
        

plt.ioff()  # disk trajector 
plt.show() # disk trajector 

plot_trajectory(disk_positions)
print("Plot disk trajectory done")

# msd_values, time_steps = compute_msd(disk_positions)
# plot_msd(msd_values, time_steps, dt)
# print("Plot msd_vs_time done")


# # Estimate the diﬀusion coeﬃcient D
# msd_values = np.array(msd_values)
# time_steps = np.array(time_steps) * dt

# linear_region = time_steps > 50

# def linear_func(t, D):
#     return 4 * D * t

# popt, pcov = curve_fit(linear_func, time_steps[linear_region], msd_values[linear_region])
# D_estimated = popt[0]

# print(f"Diﬀusion coeﬃcient  D = {D_estimated}")
