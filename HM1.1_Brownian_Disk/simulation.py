import numpy as np
from objects import Particle, Disk

# Initialize the neighbour list.
# def list_neighbours(disk, particles,  cutoff_radius):
#     '''Prepare a neigbours list for the disk.'''
    
#     neighbours = []
#     neighbour_number = []
    
#     for particle in particles:
#         distances = np.linalg.norm(particle.position - disk.position) - disk.radius 
#         if distance <= cutoff_radius
#             neighbour 
#         neighbor_indices = np.where(distances <= cutoff_radius)
#         neighbours.append(neighbor_indices)
#         neighbour_number.append(len(neighbor_indices))
#         return neighbours, neighbour_number


def initialize_particles(N_part, box_size, v0, m):
    particles = []
    spacing = box_size / np.sqrt(N_part) 

    grid_x = int(np.sqrt(N_part))
    grid_y = int(np.sqrt(N_part))

    positions = []
    for i in range(grid_x):
        for j in range(grid_y): # put particle to the center of squared lattice
            x = (i + 0.5) * spacing 
            y = (j + 0.5) * spacing
            positions.append([x, y])

    for i in range(N_part):
        position = positions[i]
        angle = np.random.uniform(0, 2 * np.pi)
        # angle of velocity

        velocity = v0 * np.array([np.cos(angle), np.sin(angle)])
        # velocity of x axis and y axis

        particle = Particle(position, velocity, m)
        particles.append(particle)
    return particles


def initialize_disk(box_size, mass_disk, radius_disk):
    position = np.array([box_size / 2, box_size / 2])
    velocity = np.array([0.0, 0.0])
    disk = Disk(position, velocity, mass_disk, radius_disk)
    return disk


def remove_overlapping_particles(particles, disk, sigma):
    filtered_particles = []
    for particle in particles:
        distance = np.linalg.norm(particle.position - disk.position) - disk.radius 
        # Calculate the Euclidean length of distance vector
        if distance >= 3 * sigma: 
            # remove particles that are closer to the disk rim
            filtered_particles.append(particle)
        else:
            pass  
    return filtered_particles


def compute_force(particle, disk, epsilon, sigma):
   
   # compute force between one  neighbour particle and the disk 
    r_vector = disk.position - particle.position 
    distance = np.linalg.norm(r_vector) # distance to disk center
    r = distance - disk.radius  # distance to the disk rim 

    force_magnitude = 24 * epsilon / r * (2 * (sigma / r)**12 - (sigma / r)**6) 
    force_vector = force_magnitude * (r_vector / distance) # disk ->> particle 
    return force_vector
