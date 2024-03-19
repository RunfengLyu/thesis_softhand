import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, max_out=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.cumulative_error = 0
        self.previous_error = 0

    def reset(self):
        self.cumulative_error = 0
        self.previous_error = 0

    def step(self, setpoint, current_value):
        error = setpoint - current_value
        self.cumulative_error += error
        derivative_error = error - self.previous_error

        output = self.kp * error + self.ki * self.cumulative_error + self.kd * derivative_error

        if self.max_out is not None:
            output = max(min(output, self.max_out), -self.max_out)

        self.previous_error = error

        return output
    
class GraspController:
    def __init__(self, model, kp=1.0, ki=0.0, kd=0.0, max_out=None):
        self.model = model
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = np.zeros(5)
        self.previous_error = np.zeros(5)

    def reset(self):
        self.integral = np.zeros(5)
        self.previous_error = np.zeros(5)

    def step(self, setpoints, current_values):
        assert len(setpoints) == len(current_values) == 5, "Expected 5 setpoints and 5 current values"
        error = setpoints - current_values
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.previous_error = error

        if self.max_out is not None:
            output = np.clip(output, -self.max_out, self.max_out)

        return output