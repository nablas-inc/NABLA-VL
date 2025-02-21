from transformers import AutoConfig, Phi3Config


class NablaVLConfig(Phi3Config):
    model_type = "nabla_vl"


AutoConfig.register("nabla_vl", NablaVLConfig)
