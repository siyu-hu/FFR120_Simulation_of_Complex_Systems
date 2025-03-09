import numpy as np
import matplotlib as plt


# Particles constant parameters 
m = 1  # Mass (units of m0).

# Disk constant parameters
disk_mass = 10
disk_radius = 10 # radius of disk

# Parameters for the simulation.
N_particles = 625  # Number of particles.
sigma = 1  # Size (units of sigma0).
epsilon = 1  # Energy (unit of epsilon0).

dt = 0.005   # Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0))).
T_total = 400
num_steps = T_total / dt

L = 260  # Box size 
x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

# neighbours for disk
cutoff_radius = 3 * disk_radius  # Cutoff_radius for neighbours list.



def list_neighbours(x, y, disk_x, disk_y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for disk'''
    neighbours = []
    neighbour_number = 0  
    
    for j in range(N_particles):
        distance = np.sqrt((disk_x - x[j]) ** 2 + (disk_y - y[j]) ** 2)
    
        if distance <= cutoff_radius:
            neighbours.append(j)  
            neighbour_number += 1

    return neighbours, neighbour_number


def single_force_cutoff(x_particle, y_particle, disk_x, disk_y, sigma, epsilon, disk_radius):
    '''
    Calculate the force on the disk due to a single particle using the Lennard-Jones potential.
    
    Parameters:
    - x_particle, y_particle: one of the neighbor particle position
    - disk_x, disk_y:  disk position 
    - sigma: Size parameter for Lennard-Jones potential.
    - epsilon: Energy parameter for Lennard-Jones potential.
    
    Returns:
    - Fx, Fy: Force on the disk.
    '''
    
  
    r2 = (disk_x - x_particle) ** 2 + (disk_y - y_particle) ** 2 
    r = np.sqrt(r2) -  disk_radius  # Distance 

    ka = sigma / r
    F = 24 * epsilon / r * (2 * (ka ** 12) - (ka ** 6))  # force magenitute
    
    Fx = F * (disk_x - x_particle) / np.sqrt(r2) # force vector :  particle to disk 
    Fy = F * (disk_y - y_particle) / np.sqrt(r2)

    return Fx, Fy 


import time
from scipy.constants import Boltzmann as kB 
from tkinter import *
import math 
import matplotlib.pyplot as plt

# Visualization 

v0 = 10  # Initial speed 
x0, y0 = np.meshgrid(
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
)
x0 = x0.flatten()[:N_particles]
y0 = y0.flatten()[:N_particles]
phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi


filtered_indices = []
for i in range(N_particles):
    distance_to_center = np.sqrt((x0[i] - 0) ** 2 + (y0[i] - 0) ** 2)
    if distance_to_center > disk_radius + 3 * sigma:
        filtered_indices.append(i)

# Filter the particle positions and angles
x0 = x0[filtered_indices]
y0 = y0[filtered_indices]
phi0 = phi0[filtered_indices]
N_particles = len(filtered_indices)  # Update the number of particles


# Preperation for particle update
x = x0
y = y0
v = np.full(N_particles, v0) # save all particles velocity 
phi = phi0
vx = v0 * np.cos(phi0)
vy = v0 * np.sin(phi0)

disk_x= 0
disk_y = 0

disk_phi = 0
disk_v = 0
disk_vx = disk_v * np.cos(disk_phi)
disk_vy = disk_v * np.sin(disk_phi)


disk_positions = []
disk_velocities = []
disk_positions.append([disk_x, disk_y])
#disk_velocities.append([disk_vx, disk_vy])
#disk_phi.append(disk_phi)

# # Plot the initial state
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)

# ax.scatter(x, y, color='cyan', s=10)

# disk_circle = plt.Circle((disk_x, disk_y), disk_radius, color='red', fill=False, linewidth=2)
# ax.add_patch(disk_circle)

# # Adding labels and title
# ax.set_title("Initial State of Particles and Disk")
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")
# ax.legend()

# # Show plot
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid()
# plt.show()

window_size = 600

tk = Tk()
tk.geometry(f'{window_size + 20}x{window_size + 20}')
tk.configure(background='#000000')

canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
tk.attributes('-topmost', 0)
canvas.place(x=10, y=10, height=window_size, width=window_size)


neighbor_particles = [] # save neighbor particles elements

particles = []
for j in range(1, N_particles):
    particles.append(
        canvas.create_oval(
            (x[j] - sigma / 2) / L * window_size + window_size / 2, 
            (y[j] - sigma / 2) / L * window_size + window_size / 2,
            (x[j] + sigma / 2) / L * window_size + window_size / 2, 
            (y[j] + sigma / 2) / L * window_size + window_size / 2,
            outline='#00C0C0', 
            fill='#00C0C0',
        )
    )


disk_element = canvas.create_oval(
    (disk_x - disk_radius) / L * window_size + window_size / 2,
    (disk_y - disk_radius) / L * window_size + window_size / 2,
    (disk_x + disk_radius) / L * window_size + window_size / 2,
    (disk_y + disk_radius) / L * window_size + window_size / 2,
    outline='#FF0000', 
    fill='',  # 可使用透明填充或特定颜色
)

step = 0

def stop_loop(event):
    global running
    running = False
tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.


# Define boundary condition  for particle 
def boundary_condition_particle(x, y, vx, vy , x_min, x_max, y_min, y_max):
    if x < x_min:
            x = x_min + (x_min - x)
            vx = - vx

    if x > x_max:
        x = x_max - (x - x_max)
        vx = - vx

    if y < y_min:
        y = y_min + (y_min - y)
        vy = - vy
            
    if y > y_max:
        y = y_max - (y - y_max)
        vy = - vy
    return x, y, vx, vy 

# Define boundary condition for disk (consider the radius)
def boundary_condition_disk(disk_radius, disk_x, disk_y, disk_vx, disk_vy , x_min, x_max, y_min, y_max):
    if disk_x - x_min <= disk_radius:
        disk_x = x_min + (x_min - disk_x)
        disk_vx = - disk_vx

    if x_max - disk_x <= disk_radius :
        disk_x = x_max - (disk_x - x_max)
        disk_vx = - disk_vx

    if disk_y - y_min <=  disk_radius:
        disk_y = y_min + (y_min - disk_y)
        disk_vy = - disk_vy
            
    if y_max - disk_y <= disk_radius:
        disk_y = y_max - (disk_y - y_max)
        disk_vy = - disk_vy

    return disk_x, disk_y, disk_vx, disk_vy 


# Main loop

running = True  # Flag to control the loop.

while running:
    neighbours, neighbour_number = list_neighbours( x, y, disk_x, disk_y, N_particles, cutoff_radius)
    # DEBUG neighbor
    # plt.figure(figsize=(6, 6))
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # disk_circle = plt.Circle((disk_x, disk_y), disk_radius, color='r', fill=False, linestyle='--', linewidth=2)
    # plt.gca().add_patch(disk_circle)

    # for i in range(N_particles):
    #     if i in neighbours:
    #        
    #         plt.scatter(x[i], y[i], color='magenta', s=10)
    #     else:
    #   
    #         plt.scatter(x[i], y[i], color='cyan', s=5, alpha=0.5)
    
    # plt.title(f'Time: {step * dt:.3f}')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # plt.pause(3)
    # plt.show()


    total_force = np.array([0.0, 0.0])
    
    half_disk_x = disk_x + disk_vx * dt * 0.5
    half_disk_y = disk_y + disk_vy * dt * 0.5


    for i in range(neighbour_number):
        half_x =0
        half_y =0

        particle_index = neighbours[i]
        half_x = x[particle_index] + vx[particle_index] * dt * 0.5
        half_y = y[particle_index] + vy[particle_index] * dt * 0.5 

        fx, fy = single_force_cutoff(half_x, half_y, half_disk_x, half_disk_y, sigma, epsilon, disk_radius)
        
        total_force[0] += fx  # Fx : total force on disk
        total_force[1] += fy  # Fy: total force on disk
        
        nvx = vx[particle_index] - fx / m * dt # N2 law, -fx = force on particle 
        nvy = vy[particle_index] - fy / m * dt # N2 law, -fy = force on particle 
        
        #print(f'half_x ={half_x} nvx = {nvx}, dt={dt}')
        nx = half_x + (nvx * dt * 0.5)
        ny = half_y + (nvy * dt * 0.5)  

        nx, nx , nvx, nvy = boundary_condition_particle(nx, ny, nvx, nvy, x_min, x_max, y_min, y_max)

        x[particle_index] = nx
        y[particle_index] = ny
        vx[particle_index] = nvx
        vy[particle_index] = nvy

        #print(f'Step time  = {step}, nei. num = {neighbour_number}, nei.index = {i}, total force = {total_force} ')
        #print(f'Particle position= [ {x[particle_index]}, {y[particle_index]}] ')

    next_disk_vx = disk_vx + total_force[0] / disk_mass  * dt
    next_disk_vy = disk_vy + total_force[1] / disk_mass  * dt

    next_disk_x = half_disk_x + next_disk_vx * dt * 0.5
    next_disk_y = half_disk_y + next_disk_vy * dt * 0.5


    next_disk_x, next_disk_y, next_disk_vx, next_disk_vy = boundary_condition_disk(disk_radius, next_disk_x, next_disk_y, next_disk_vx, next_disk_vy, x_min, x_max, y_min, y_max )

    disk_x = next_disk_x
    disk_y = next_disk_y
    disk_vx = next_disk_vx
    disk_vy = next_disk_vy
  


    disk_positions.append([disk_x, disk_y])

    # update particle that not in neighbors
    for j in range(N_particles):
        if j not in neighbours:
            x[j] += vx[j] * dt
            y[j] += vy[j] * dt
            x[j], y[j], vx[j], vy[j] = boundary_condition_particle(x[j], y[j], vx[j], vy[j], x_min, x_max, y_min, y_max)

  
    if step % 100 == 0:
        print(f'Steps = {step}, total force on disk ={total_force}, disk position = [{disk_x}, {disk_y}]')
        print(f'neighbor index list: {neighbours}, number of nei. ={neighbour_number}') 
        

    # # Update neighbors
    # neighbours, neighbour_number = list_neighbours(x, y, disk_x, disk_y, N_particles, cutoff_radius)
    # print(f'neighbor index list: {neighbours}, number of nei. ={neighbour_number}') 
    # # clear previous neighbor elements
    # for neighbor_particle in neighbor_particles:
    #     canvas.delete(neighbor_particle)
    #     neighbor_particles.clear()

    # # Draw new neighbors
    # for particle_index in neighbours:
    #     neighbor_particles.append(
    #         canvas.create_oval(
    #             (x[particle_index] - sigma / 2) / L * window_size + window_size / 2,
    #             (y[particle_index] - sigma / 2) / L * window_size + window_size / 2,
    #             (x[particle_index] + sigma / 2) / L * window_size + window_size / 2,
    #             (y[particle_index] + sigma / 2) / L * window_size + window_size / 2,
    #             outline='#FF00FF',  
    #              fill='#FF00FF',
    #         )
    #     )
 
    # Update animation frame.
    if step % 100 == 0:        

        for j, particle in enumerate(particles):
            canvas.coords(
                particle,
                (x[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (y[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (x[j + 1] + sigma / 2) / L * window_size + window_size / 2,
                (y[j + 1] + sigma / 2) / L * window_size + window_size / 2,
            )
        # Update disk visualization (assuming there is a disk element created)
        canvas.coords(
            disk_element,
            (disk_x - disk_radius) / L * window_size + window_size / 2,
            (disk_y - disk_radius) / L * window_size + window_size / 2,
            (disk_x + disk_radius) / L * window_size + window_size / 2,
            (disk_y + disk_radius) / L * window_size + window_size / 2,
        )
                    
        tk.title(f'Time {step * dt:.1f} - Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(.001)  # Increase to slow down the simulation.    

    step += 1


    if step >= num_steps:
        running = False
        tk.destroy() 
    

# Plot disk trajactory 
        disk_positions = np.array(disk_positions)
        x_positions = disk_positions[:, 0]
        y_positions = disk_positions[:, 1]
        plt.figure(figsize=(10, 6))
        plt.plot(x_positions, y_positions, label="Disk Position Trajectory", color='blue', linestyle='-', marker='o', markersize=2)
        plt.title("Disk Position Trajectory Over Time")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend()
        plt.show()

tk.update_idletasks()
tk.update()
tk.mainloop()
