import torch
import gc
from lama_cleaner.helper import switch_mps_device
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config

models = {
    "lama": LaMa,
}


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.model = self.init_model(name, device, **kwargs)

    def init_model(self, name: str, device, **kwargs):
        if name in models:
            model = models[name](device, **kwargs)
        else:
            raise NotImplementedError(f"Not supported model: {name}")
        return model

    def is_downloaded(self, name: str) -> bool:
        if name in models:
            return models[name].is_downloaded()
        else:
            raise NotImplementedError(f"Not supported model: {name}")

    def __call__(self, image, mask, config: Config):
        self.switch_controlnet_method(control_method=config.controlnet_method)
        return self.model(image, mask, config)

    def switch(self, new_name: str, **kwargs):
        if new_name == self.name:
            return
        try:
            if torch.cuda.memory_allocated() > 0:
                # Clear current loaded model from memory
                torch.cuda.empty_cache()
                del self.model
                gc.collect()

            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
            self.name = new_name
        except NotImplementedError as e:
            raise e

    def torch_gc():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    def switch_controlnet_method(self, control_method: str):
        if not self.kwargs.get("sd_controlnet"):
            return
        if self.kwargs["sd_controlnet_method"] == control_method:
            return
        if not hasattr(self.model, "is_local_sd_model"):
            return


        del self.model
        torch_gc()

        old_method = self.kwargs["sd_controlnet_method"]
        self.kwargs["sd_controlnet_method"] = control_method
        self.model = self.init_model(
            self.name, switch_mps_device(self.name, self.device), **self.kwargs
        )
