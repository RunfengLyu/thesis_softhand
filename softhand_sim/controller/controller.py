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