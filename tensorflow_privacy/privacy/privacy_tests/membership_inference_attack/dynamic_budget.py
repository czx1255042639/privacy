# tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/dynamic_budget.py
import numpy as np

class DynamicPrivacyScheduler:
    def __init__(self, 
                 initial_epsilon=5.0, 
                 warmup_epochs=5, 
                 decay_rate=0.1, 
                 epsilon_min=0.5,
                 sensitivity=1.0):
        self.initial_epsilon = initial_epsilon
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.sensitivity = sensitivity
        self.initial_loss_curvature = None

    def get_noise_multiplier(self, epoch, current_loss_curvature):
        """根据当前epoch和损失曲率计算噪声乘数"""
        if epoch < self.warmup_epochs:
            epsilon = self.initial_epsilon
        else:
            if self.initial_loss_curvature is None:
                self.initial_loss_curvature = current_loss_curvature + 1e-8
            decay_factor = 1 - (current_loss_curvature / self.initial_loss_curvature)
            epsilon = self.initial_epsilon * decay_factor * np.exp(-self.decay_rate * (epoch - self.warmup_epochs))
            epsilon = max(epsilon, self.epsilon_min)
        
        # 噪声乘数 = 敏感度 / epsilon
        return self.sensitivity / epsilon
