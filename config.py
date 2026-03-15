class Configuration:
    def __init__(self,
            run_name: str,
            part: float,
            message_dim: int,
            md_opt_decode: bool,
            md_opt_watermark: bool,
            epochs: int,
            test_images_num: int,
            enable_quantization_noise: bool,
            noise_level: int,
            message_recovery_loss_weight: float,
            learning_rate: float,
        ):
        self.run_name = run_name
        self.part = part
        self.message_dim = message_dim
        self.md_opt_decode = md_opt_decode
        self.md_opt_watermark = md_opt_watermark
        self.epochs = epochs
        self.test_images_num = test_images_num
        self.enable_quantization_noise = enable_quantization_noise
        self.noise_level = noise_level
        self.message_recovery_loss_weight = message_recovery_loss_weight
        self.learning_rate = learning_rate
class InferenceConfiguration:
    """Configuration for inference (watermark encoding) only. Does not affect train.py."""

    def __init__(self, message_dim: int = 128):
        self.message_dim = message_dim