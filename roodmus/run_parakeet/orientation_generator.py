import numpy as np

### class to generate orientations for a given molecule based on some specifications
class orientation_generator(object):
    def __init__(self):
        pass

    @classmethod       
    def generate_inplane(self, n=1):
        poses = []
        for i in range(n):
            phi = 0
            theta = 0
            psi = np.random.uniform(0, 2*np.pi)
            poses.append((phi, theta, psi))
        return poses
        
