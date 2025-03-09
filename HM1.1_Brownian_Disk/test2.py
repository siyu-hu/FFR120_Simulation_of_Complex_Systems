import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tkinter import *

# Simulation parameters
m = 1  # Particle mass
disk_mass = 10

N_particles = 625
sigma = 1
epsilon = 1
dt = 0.005
T_tot = 400
L = 260  # Box size
x_min, x_max, y_min, y_max = -L / 2, L / 2, -L / 2, L / 2
R_disk = 10
output_folder = "./"  # Specify the folder to save the results

# Initialize particles and disk
particles = [{'x': np.random.uniform(-L / 2, L / 2), 'y': np.random.uniform(-L / 2, L / 2),
              'vx': np.random.uniform(-1, 1), 'vy': np.random.uniform(-1, 1)} for _ in range(N_particles)]
disk = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}

# Tkinter setup for animation
window_size = 600
window = Tk()
window.geometry(f"{window_size + 20}x{window_size + 20}")
canvas = Canvas(window, background="#ECECEC")
canvas.place(x=10, y=10, height=window_size, width=window_size)
particle_objs = [canvas.create_oval(p['x'], p['y'], p['x'] + 2, p['y'] + 2, fill="cyan") for p in particles]
disk_obj = canvas.create_oval(disk['x'], disk['y'], disk['x'] + R_disk, disk['y'] + R_disk, outline="red")

# Initialize lists for trajectory and MSD calculations
trajectory = []
msd_list = []
time_steps = int(T_tot / dt)
step = 0
running = True
# Main loop

running = True
step = 0

while running and step < time_steps:
    # Record disk's position for trajectory and MSD calculation
    trajectory.append((disk['x'], disk['y']))

    # Calculate MSD
    if step > 0:
        initial_x, initial_y = trajectory[0]
        msd = (disk['x'] - initial_x) ** 2 + (disk['y'] - initial_y) ** 2
        msd_list.append(msd)

    # Calculate forces and update positions
    disk_fx, disk_fy = 0, 0
    for i, particle in enumerate(particles):
        dx = disk['x'] - particle['x']
        dy = disk['y'] - particle['y']
        r = np.sqrt(dx ** 2 + dy ** 2) - R_disk
        if r < 3 * sigma:
            F = 24 * epsilon * ((2 * (sigma / r) ** 12) - (sigma / r) ** 6) / r
            fx = F * dx / r
            fy = F * dy / r
            disk_fx += fx
            disk_fy += fy
            particle['vx'] += -fx / m * dt
            particle['vy'] += -fy / m * dt
        particle['x'] += particle['vx'] * dt
        particle['y'] += particle['vy'] * dt

        # Particle boundary conditions
        if particle['x'] <= 0 or particle['x'] >= L:
            particle['vx'] *= -1
        if particle['y'] <= 0 or particle['y'] >= L:
            particle['vy'] *= -1

        # Update particle position in Tkinter
        canvas.coords(particle_objs[i], particle['x'] - 2, particle['y'] - 2, particle['x'] + 2, particle['y'] + 2)

    # Update disk velocity and position based on net force
    disk['vx'] += disk_fx / disk_mass * dt
    disk['vy'] += disk_fy / disk_mass * dt
    disk['x'] += disk['vx'] * dt
    disk['y'] += disk['vy'] * dt

    # Disk boundary conditions
    if disk['x'] <= R_disk or disk['x'] >= L - R_disk:
        disk['vx'] *= -1
    if disk['y'] <= R_disk or disk['y'] >= L - R_disk:
        disk['vy'] *= -1

    # Update disk position in Tkinter
    canvas.coords(disk_obj, disk['x'] - R_disk, disk['y'] - R_disk, disk['x'] + R_disk, disk['y'] + R_disk)
    canvas.create_oval(disk['x'] - 0.2, disk['y'] - 0.2, disk['x'] + 0.2, disk['y'] + 0.2, fill="red", outline="red")

    if step % 100 == 0:
        print(f"Step: {step}, Disk Position: ({disk['x']:.2f}, {disk['y']:.2f})")

    # Update animation frame
    window.update_idletasks()  # 使用 window 而不是 tk
    window.update()            # 使用 window 而不是 tk
    time.sleep(0.001)
    step += 1

# Plot the trajectory and MSD after the loop
trajectory_x, trajectory_y = zip(*trajectory)
plt.figure(figsize=(6, 5))
plt.plot(trajectory_x, trajectory_y, color='blue')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Disk Trajectory in Cartesian Plane")
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'disk_trajectory.png'))
plt.show()

# Plot MSD
plt.figure(figsize=(6, 5))
time_values = np.arange(1, len(msd_list) + 1) * dt
plt.plot(time_values, msd_list, color='red')
plt.xlabel("Time (t)")
plt.ylabel("Mean Square Displacement (MSD)")
plt.title("Mean Square Displacement over Time")
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'msd_over_time.png'))
plt.show()

# Close Tkinter window
window.destroy()
