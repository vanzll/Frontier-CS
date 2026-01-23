import math
import numpy as np

class LunarLanderEnv:
    """
    Simplified LunarLander dynamics (No Box2D dependency).
    
    Goal: Land on the landing pad at (0,0).
    
    State (8D):
        0: x (horizontal position)
        1: y (vertical position)
        2: vx (horizontal velocity)
        3: vy (vertical velocity)
        4: theta (angle)
        5: vtheta (angular velocity)
        6: left_leg_contact (bool, 1.0 or 0.0)
        7: right_leg_contact (bool, 1.0 or 0.0)
        
    Action (Discrete 4):
        0: Do nothing
        1: Fire left orientation engine
        2: Fire main engine
        3: Fire right orientation engine
    """
    def __init__(self):
        self.dt = 0.04
        self.gravity = -3.0 # Moon gravity is less than Earth
        self.main_engine_power = 13.0
        self.side_engine_power = 0.6
        
        self.initial_random = 1000.0  # Force applied on reset
        
        self.state = None
        self.fuel_spent = 0.0
        
        # Thresholds
        self.max_y = 2.0
        self.min_y = 0.0 # Ground
        self.max_x = 1.5
        self.min_x = -1.5
        
    def reset(self):
        # Start high up
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(1.0, 1.4)
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.0) # Downward velocity
        theta = np.random.uniform(-0.1, 0.1)
        vtheta = np.random.uniform(-0.1, 0.1)
        
        self.state = np.array([x, y, vx, vy, theta, vtheta, 0.0, 0.0], dtype=np.float32)
        self.fuel_spent = 0.0
        return self.state

    def step(self, action):
        x, y, vx, vy, theta, vtheta, l_contact, r_contact = self.state
        
        # Forces
        acc_x = 0.0
        acc_y = self.gravity
        acc_theta = 0.0
        
        m_power = 0.0
        s_power = 0.0
        
        # Engines
        if action == 2: # Main engine
            m_power = 1.0
            acc_x += math.sin(theta) * self.main_engine_power * 0.5 # Scale down for stability
            acc_y += math.cos(theta) * self.main_engine_power * 0.5
            
        elif action == 1: # Left engine (pushes right)
            s_power = 1.0
            acc_theta -= self.side_engine_power
            
        elif action == 3: # Right engine (pushes left)
            s_power = 1.0
            acc_theta += self.side_engine_power
            
        # Update physics
        x += vx * self.dt
        y += vy * self.dt
        theta += vtheta * self.dt
        
        vx += acc_x * self.dt
        vy += acc_y * self.dt
        vtheta += acc_theta * self.dt
        
        # Collision / Ground detection
        done = False
        landed = False
        crashed = False
        
        if y <= 0.0:
            done = True
            y = 0.0
            # Check landing condition
            # Must be upright, slow, and near center
            if (abs(x) < 0.2 and abs(theta) < 0.2 and 
                abs(vy) < 0.5 and abs(vx) < 0.5):
                landed = True
            else:
                crashed = True
        
        # Out of bounds
        if abs(x) > 1.5 or y > 2.5:
            done = True
            crashed = True
            
        self.state = np.array([x, y, vx, vy, theta, vtheta, 0.0, 0.0], dtype=np.float32)
        
        # Ground Truth Reward (Similar to standard Gym LunarLander)
        # - shaping based on distance/speed/angle
        # +100 for landing, -100 for crash
        # -0.3 per frame for main engine (fuel)
        # -0.03 per frame for side engine
        
        # We compute this to emulate the "True" reward users aim for
        # Distance penalty
        dist = math.sqrt(x*x + y*y)
        vel = math.sqrt(vx*vx + vy*vy)
        
        reward = 0.0
        # This shaping is what standard LunarLander does internally.
        # But we want the user to DESIGN this.
        # So the "Ground Truth" reward for evaluation should be simple:
        # +100 for Landing, -100 for Crash. 
        # Fuel cost is real too.
        
        if crashed:
            reward = -100.0
        elif landed:
            reward = 100.0
            
        # Fuel cost is always there in ground truth
        reward -= m_power * 0.3 * self.dt # Scaled by dt? No, standard is per step
        reward -= s_power * 0.03 * self.dt
        
        return self.state, reward, done, False, {}

    @property
    def observation_space_shape(self):
        return (8,)
    
    @property
    def action_space_n(self):
        return 4

