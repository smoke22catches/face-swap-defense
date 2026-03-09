class Configuration:
    def __init__(self,
            run_name: str,
            part: float,
            message_dim: int,
            md_opt_decode: bool,
            md_opt_watermark: bool,
        ):
        self.run_name = run_name
        self.part = part
        self.message_dim = message_dim
        self.md_opt_decode = md_opt_decode
        self.md_opt_watermark = md_opt_watermark