import numpy as np

class Particle:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass

    def apply_boundary_conditions(self, box_size):
        # when collide to boundary of box, paticle reflects to inside with negative of velocity
        # Y axis
        if self.position[0] < 0:  # left
            self.position[0] = -self.position[0] 
            self.velocity[0] = -self.velocity[0] 
        elif self.position[0] > box_size:  # right
            self.position[0] = 2 * box_size - self.position[0] 
            self.velocity[0] = -self.velocity[0]  

        # X aixs
        if self.position[1] < 0:  # Down
            self.position[1] = 0 
            self.velocity[1] = -self.velocity[1]  
        elif self.position[1] > box_size:  # Up
            self.position[1] = 2 * box_size - self.position[1]
            self.velocity[1] = -self.velocity[1]  
            

class Disk:
    def __init__(self, position, velocity, mass, radius):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius

    def apply_boundary_conditions(self, box_size):
        # when collide to boundary of box, paticle reflects to inside with negative of velocity(on x aixs or y axis)
        # Y axis
        if self.position[0] - self.radius < 0:  # left
            self.position[0] = self.radius + (self.radius - self.position[0])     
            self.velocity[0] = -self.velocity[0]  
        elif self.position[0] + self.radius > box_size:  # right
            self.position[0] = (2 * box_size - self.position[0]) - self.radius   
            self.velocity[0] = -self.velocity[0] 
        # X axis
        if self.position[1] - self.radius < 0:  # down
            self.position[1] = self.radius + (self.radius - self.position[1])    
            self.velocity[1] = -self.velocity[1]  
        elif self.position[1] + self.radius > box_size:  # Up
            self.position[1] = (2 * box_size - self.position[1]) - self.radius    
            self.velocity[1] = -self.velocity[1]  
