import numpy as np
import control as ctrl
import collections

def actuator_model(u_desired, u_ac_prev, dt, RSat, Sat):
    """
    Apply actuator dynamics with rate limiting and saturation
    
    Args:
        u_desired: Desired control input
        u_ac_prev: Previous actuator output
        dt: Time step
        RSat: Rate saturation limit
        Sat: Magnitude saturation limit
        
    Returns:
        u_ac_new: New actuator output
    """
    delta_u = u_desired - u_ac_prev
    if delta_u > RSat * dt:
        u_ac_new = u_ac_prev + RSat * dt
    elif delta_u < -RSat * dt:
        u_ac_new = u_ac_prev - RSat * dt
    else:
        u_ac_new = u_desired
    u_ac_new = np.clip(u_ac_new, -Sat, Sat)
    return u_ac_new

def apply_uncertainty(u_in, control_buffer, gain, delay_steps):
    """
    Apply gain and delay uncertainty to control input
    
    Args:
        u_in: Input control signal
        control_buffer: Buffer for delayed signal (deque)
        gain: Input gain multiplier
        delay_steps: Number of delay steps
        
    Returns:
        u_out: Output control signal with uncertainty
    """
    gained = u_in * gain
    control_buffer.append(gained)
    if delay_steps > 0:
        delayed = control_buffer.popleft()
        return delayed
    else:
        return gained

class EverythingEnv:
    def __init__(
        self,
        dt=0.01,
        TOTAL_TIME=25.0,
        max_action=20.0,
        reference=1.0,
        reference_type='constant',
        disturbance_time=15.0,
        disturbance_value=0.0,
        noise_time=20.0,
        noise_std=0.0,
        use_actuator=True,
        Sat=20.0,
        RSat=30.0,
        use_uncertainty=True,
        gain=1.0,
        delay=0.42,
        sine_amplitude=1.0,
        sine_freq=1.0,
        step_value=1.0,
        control_mode="ddpg"
    ):
        # System parameters
        self.dt = dt
        self.TOTAL_TIME = TOTAL_TIME
        self.MAX_STEPS = int(self.TOTAL_TIME / self.dt)
        self.max_action = max_action
        
        # Set AUV yaw motion model (3rd order linear system)
        A = np.array([
            [0,     1,      0],
            [0,     0,      1],
            [0, -5.375, -5.235]
        ])
        B = np.array([
            [0],
            [1.816],
            [-3.770]
        ])
        C = np.array([[1, 0, 0]])
        D = np.array([[0]])
        
        # Create continuous and discrete system models
        self.sys_c = ctrl.ss(A, B, C, D)
        self.sys_d = ctrl.c2d(self.sys_c, self.dt)
        
        # Current state initialization
        self.x = np.zeros((3, 1))
        self.prev_error = 0
        self.integral_error = 0
        self.time = 0
        self.steps = 0
        
        # Reference settings
        self.reference_type = reference_type
        self.reference = reference
        self.sine_amplitude = sine_amplitude
        self.sine_freq = sine_freq
        self.step_value = step_value
        
        # Disturbance and noise settings
        self.disturbance_time = disturbance_time
        self.disturbance_value = disturbance_value
        self.noise_time = noise_time
        self.noise_std = noise_std
        
        # Actuator dynamics settings
        self.use_actuator = use_actuator
        self.u_ac_prev = 0
        self.Sat = Sat
        self.RSat = RSat
        
        # Uncertainty settings
        self.use_uncertainty = use_uncertainty
        self.gain = gain
        self.delay = delay
        self.delay_steps = int(np.ceil(self.delay / self.dt))
        self.control_buffer = collections.deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        # Control mode
        self.control_mode = control_mode
            
        # Define state and action dimensions
        self.state_dim = 4  # [output, error, derivative_error, integral_error]
        self.action_dim = 1
    
    def get_reference(self):
        """Get the current reference value based on time and reference type"""
        if self.reference_type == 'constant':
            return self.reference
        elif self.reference_type == 'sine':
            return self.sine_amplitude * np.sin(self.sine_freq * self.time)
        elif self.reference_type == 'step':
            return self.step_value
        elif self.reference_type == 'custom_step':
            # Custom step pattern: 1 (0-10s), 0 (10-20s), -1 (20-30s), 0 (30-40s), 1 (40+s)
            if self.time < 10.0:
                return 1.0
            elif self.time < 20.0:
                return 0.0
            elif self.time < 30.0:
                return -1.0
            elif self.time < 40.0:
                return 0.0
            else:
                return 1.0
        else:
            return self.reference  # Default to constant reference
    
    def reset(self):
        # Reset state and time variables
        self.x = np.zeros((3, 1))
        self.prev_error = 0
        self.integral_error = 0
        self.time = 0
        self.steps = 0
        
        # Reset actuator state
        self.u_ac_prev = 0
        
        # Reset delay buffer
        self.control_buffer = collections.deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
            
        # Get current reference
        current_reference = self.get_reference()
        
        # Compute initial state representation
        y = self.x[0, 0]
        error = current_reference - y
        derivative_error = 0
        integral_error = 0
        
        # Ensure state is a flat array with shape (4,)
        state = np.array([y, error, derivative_error, integral_error], dtype=np.float32).flatten()
        return state
    
    def step(self, action):
        # Clip action to allowed range
        raw_action = np.clip(action, -self.max_action, self.max_action)
        
        # Apply actuator dynamics if enabled
        if self.use_actuator:
            u_ac = actuator_model(raw_action, self.u_ac_prev, self.dt, self.RSat, self.Sat)
            self.u_ac_prev = u_ac
        else:
            u_ac = raw_action
        
        # Apply input uncertainty if enabled
        if self.use_uncertainty:
            u_final = apply_uncertainty(u_ac, self.control_buffer, self.gain, self.delay_steps)
        else:
            u_final = u_ac
        
        # Apply the control input to the system
        u = np.array([[u_final]])
        
        # Update system state
        self.x = self.sys_d.A @ self.x + self.sys_d.B @ u
        
        # Get system output
        y_true = (self.sys_d.C @ self.x)[0, 0]
        
        # Apply disturbance if time exceeds disturbance_time
        if self.time >= self.disturbance_time:
            y_true += self.disturbance_value * self.dt
        
        # Apply measurement noise if time exceeds noise_time
        if self.time >= self.noise_time and self.noise_std > 0:
            y_true += np.random.normal(0, self.noise_std)
        
        # Get current reference
        current_reference = self.get_reference()
        
        # Calculate error metrics
        error = current_reference - y_true
        derivative_error = (error - self.prev_error) / self.dt
        self.integral_error += error * self.dt
        integral_error = self.integral_error
        
        # Update for next step
        self.prev_error = error
        self.time += self.dt
        self.steps += 1
        
        # Check if episode is done
        done = self.steps >= self.MAX_STEPS
        
        # Compute reward (penalize error and control effort)
        reward = -(10.0 * (error**2) + 1.0 * (raw_action**2))
        
        # Define state representation - ensure it's a flat array of shape (4,)
        state = np.array([y_true, error, derivative_error, integral_error], dtype=np.float32).flatten()
        
        # Return extra information for monitoring
        info = {
            'time': self.time,
            'output': y_true,
            'reference': current_reference,
            'control': raw_action,
            'u_c': raw_action,    # Raw control command for consistency with tt.py
            'u_ac': u_ac,         # After actuator dynamics
            'u_un': u_final       # After uncertainty
        }
        
        return state, reward, done, info