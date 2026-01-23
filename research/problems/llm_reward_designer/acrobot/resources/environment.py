import math
import numpy as np

class AcrobotEnv:
    """
    Manual implementation of Acrobot-v1 dynamics.
    Double pendulum, actuated at the second joint.
    
    State:
        [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    
    Action:
        0: Torque -1
        1: Torque 0
        2: Torque +1
    """
    def __init__(self):
        self.dt = 0.2
        self.LINK_LENGTH_1 = 1.0  # [m]
        self.LINK_LENGTH_2 = 1.0  # [m]
        self.LINK_MASS_1 = 1.0  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.0  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
        self.LINK_MOI = 1.0  #: moments of inertia for both links
        self.MAX_VEL_1 = 4 * math.pi
        self.MAX_VEL_2 = 9 * math.pi
        self.AVAIL_TORQUE = [-1.0, 0.0, +1]
        self.torque_noise_max = 0.0

        self.state = None # theta1, theta2, theta1_dot, theta2_dot

    def reset(self):
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def _get_ob(self):
        s = self.state
        return np.array([
            np.cos(s[0]), np.sin(s[0]), 
            np.cos(s[1]), np.sin(s[1]), 
            s[2], s[3]
        ], dtype=np.float32)

    def step(self, action):
        s = self.state
        torque = self.AVAIL_TORQUE[action]

        # Dynamics (RK4 integration would be better but keeping simple Euler for readability/speed)
        # Using the equations from Gym implementation source
        
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        
        d1 = self.LINK_MASS_1 * self.LINK_COM_POS_1 ** 2 + self.LINK_MASS_2 * (self.LINK_LENGTH_1 ** 2 + self.LINK_COM_POS_2 ** 2 + 2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2 * math.cos(theta2)) + 2 * self.LINK_MOI
        d2 = self.LINK_MASS_2 * (self.LINK_COM_POS_2 ** 2 + self.LINK_LENGTH_1 * self.LINK_COM_POS_2 * math.cos(theta2)) + self.LINK_MOI
        phi2 = self.LINK_MASS_2 * self.LINK_COM_POS_2 * 9.8 * math.cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = - self.LINK_MASS_2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2 * dtheta2 ** 2 * math.sin(theta2) - 2 * self.LINK_MASS_2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2 * dtheta2 * dtheta1 * math.sin(theta2) + (self.LINK_MASS_1 * self.LINK_COM_POS_1 + self.LINK_MASS_2 * self.LINK_LENGTH_1) * 9.8 * math.cos(theta1 - math.pi / 2.0) + phi2
        
        # Accelerations
        ddtheta2 = (torque + d2 / d1 * phi1 - self.LINK_MASS_2 * self.LINK_LENGTH_1 * self.LINK_COM_POS_2 * dtheta1 ** 2 * math.sin(theta2) - phi2) / (self.LINK_MASS_2 * self.LINK_COM_POS_2 ** 2 + self.LINK_MOI - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        
        # Update
        theta1 = theta1 + dtheta1 * self.dt
        theta2 = theta2 + dtheta2 * self.dt
        dtheta1 = dtheta1 + ddtheta1 * self.dt
        dtheta2 = dtheta2 + ddtheta2 * self.dt
        
        # Clip velocities
        dtheta1 = np.clip(dtheta1, -self.MAX_VEL_1, self.MAX_VEL_1)
        dtheta2 = np.clip(dtheta2, -self.MAX_VEL_2, self.MAX_VEL_2)
        
        self.state = np.array([theta1, theta2, dtheta1, dtheta2])
        
        # Termination: Tip is above the line
        terminated = bool(-math.cos(theta1) - math.cos(theta1 + theta2) > 1.0)
        
        # Ground Truth Reward
        # -1 per step until success, then 0.
        if not terminated:
            reward = -1.0
        else:
            reward = 0.0
            
        return self._get_ob(), reward, terminated, False, {}

    @property
    def observation_space_shape(self):
        return (6,)
    
    @property
    def action_space_n(self):
        return 3

