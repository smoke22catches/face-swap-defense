class Configuration:
    def __init__(self,
            run_name: str,
            part: float,
            message_dim: int,
            epochs: int,
            test_images_num: int,
            enable_quantization_noise: bool,
            noise_level: int,
            message_recovery_loss_weight: float,
            learning_rate: float,
            enable_diffjpeg: bool,
            jpeg_quality: int,
        ):
        self.run_name = run_name
        self.part = part
        self.message_dim = message_dim
        self.epochs = epochs
        self.test_images_num = test_images_num
        self.enable_quantization_noise = enable_quantization_noise
        self.noise_level = noise_level
        self.message_recovery_loss_weight = message_recovery_loss_weight
        self.learning_rate = learning_rate
        self.enable_diffjpeg = enable_diffjpeg
        self.jpeg_quality = jpeg_quality
class InferenceConfiguration:
    """Configuration for inference (watermark encoding) only. Does not affect train.py."""

    def __init__(self, message_dim: int = 128):
        self.message_dim = message_dim