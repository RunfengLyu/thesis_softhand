class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def calculate(self, setpoint, feedback, dt):
        error = setpoint - feedback
        self.error_sum += error * dt
        error_diff = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff

        self.prev_error = error

        return output