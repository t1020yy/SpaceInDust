import numpy as np
class ModelingParabolaParameters:

    def __init__(self):
        self.particle_diameter = 0.05 #mm
        self.x_start_trajectory = -15 #mm
        self.y_start_trajectory = 10 #mm
        self.start_speed = -0.6 * 10**3
        self.start_angle = 75 / 180 * np.pi
        self.plane_parameter_A = 0 
        self.plane_parameter_B = 0
        self.plane_parameter_C = 1
        self.interval_time = 0.0001
        self.x_integration_step = 5*10**-3
        self.y_integration_step = 5*10**-3
        self.F = -43
        self.theta = 0

