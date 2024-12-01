import os

from sympy.stats.sampling.sample_numpy import numpy

from comfy import model_management
import folder_paths
from diffusers import UniPCMultistepScheduler

from comfy.utils import ProgressBar
from folder_paths import supported_pt_extensions
from .libs.ref_encoder.adapter import *
from .libs.ref_encoder.latent_controlnet import ControlNetModel
from .libs.ref_encoder.reference_unet import RefHairUnet
from .libs.utils.pipeline import StableHairPipeline
from .libs.utils.pipeline_cn import StableDiffusionControlNetPipeline

deviceType = model_management.get_torch_device().type
current_dir = os.path.dirname(os.path.abspath(__file__))
sd_config_dir = os.path.join(current_dir, "libs/configs/sd15")
hair_model_path_format = 'StableHair/{}'


class LoadStableHairRemoverModel:

    @classmethod
    def INPUT_TYPES(cls):
        model_paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            stable_hair_path = os.path.join(search_path, "StableHair")
            if os.path.exists(stable_hair_path):
                for root, subdir, files in os.walk(stable_hair_path, followlinks=True):
                    for file in files:
                        file_name_ext = file.split(".")
                        if len(file_name_ext) > 1 and '.{}'.format(file_name_ext[-1]) in supported_pt_extensions:
                            model_paths.append(file)
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                              {"tooltip": "The name of the checkpoint (model) to load."}),
                "bald_model": (model_paths, {}),
                "device": (["AUTO", "CPU"],)
            }
        }

    RETURN_TYPES = ("BALD_MODEL",)
    RETURN_NAMES = ("bald_model",)
    FUNCTION = "load_model"
    CATEGORY = "hair/transfer"

    def load_model(self, ckpt_name, bald_model, device):
        model_management.soft_empty_cache()
        sd15_model_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        bald_model_path = folder_paths.get_full_path_or_raise("diffusers", hair_model_path_format.format(bald_model))
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"

        weight_dtype = torch.float16 if device_type == "cuda" else torch.float32

        remove_hair_pipeline = StableDiffusionControlNetPipeline.from_single_file(sd15_model_path,
                                                                                  config=sd_config_dir,
                                                                                  torch_dtype=weight_dtype,
                                                                                  local_files_only=True,
                                                                                  safety_checker=None,
                                                                                  requires_safety_checker=False,
                                                                                  )
        bald_converter = ControlNetModel.from_unet(remove_hair_pipeline.unet)
        _state_dict = torch.load(bald_model_path)

        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(device_type, dtype=weight_dtype)
        remove_hair_pipeline.register_modules(controlnet=bald_converter)

        remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(remove_hair_pipeline.scheduler.config)
        remove_hair_pipeline.to(device_type)

        return remove_hair_pipeline,


class LoadStableHairTransferModel:

    @classmethod
    def INPUT_TYPES(cls):
        model_paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            stable_hair_path = os.path.join(search_path, "StableHair")
            if os.path.exists(stable_hair_path):
                for root, subdir, files in os.walk(stable_hair_path, followlinks=True):
                    for file in files:
                        file_name_ext = file.split(".")
                        if len(file_name_ext) >1 and '.{}'.format(file_name_ext[-1]) in supported_pt_extensions:
                            model_paths.append(file)
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                              {"tooltip": "The name of the checkpoint (model) to load."}),
                "encoder_model": (model_paths, {}),
                "adapter_model": (model_paths, {}),
                "control_model": (model_paths, {}),
                "device": (["AUTO", "CPU"],)
            }
        }

    RETURN_TYPES = ("HAIR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "hair/transfer"

    def load_model(self, ckpt_name, encoder_model, adapter_model, control_model, device):
        model_management.soft_empty_cache()
        sd15_model_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        encoder_model_path = folder_paths.get_full_path_or_raise("diffusers",
                                                                 hair_model_path_format.format(encoder_model))
        adapter_model_path = folder_paths.get_full_path_or_raise("diffusers",
                                                                 hair_model_path_format.format(adapter_model))
        control_model_path = folder_paths.get_full_path_or_raise("diffusers",
                                                                 hair_model_path_format.format(control_model))
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"

        weight_dtype = torch.float16 if device_type == "cuda" else torch.float32

        pipeline = StableHairPipeline.from_single_file(sd15_model_path,
                                                       config=sd_config_dir,
                                                       torch_dtype=weight_dtype,
                                                       local_files_only=True,
                                                       safety_checker=None,
                                                       requires_safety_checker=False,
                                                       ).to(device_type)

        controlnet = ControlNetModel.from_unet(pipeline.unet).to(device_type)
        _state_dict = torch.load(control_model_path)
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(device_type, dtype=weight_dtype)
        pipeline.register_modules(controlnet=controlnet)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        hair_encoder = RefHairUnet.from_config(pipeline.unet.config)
        _state_dict = torch.load(encoder_model_path)
        hair_encoder.load_state_dict(_state_dict, strict=False)
        hair_encoder.to(device_type, dtype=weight_dtype)
        pipeline.register_modules(reference_encoder=hair_encoder)

        hair_adapter = adapter_injection(pipeline.unet, device=device_type, dtype=weight_dtype, use_resampler=False)
        _state_dict = torch.load(adapter_model_path)

        hair_adapter.load_state_dict(_state_dict, strict=False)

        return pipeline,


class ApplyHairRemover:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bald_model": ("BALD_MODEL",),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "hair/transfer"

    def apply(self, bald_model, images, seed, steps, strength, cfg=1.5):
        _images = []
        _masks = []

        for image in images:
            # h, w, c -> c, h, w
            H, W, c = image.shape
            im_tensor = image.permute(2, 0, 1)
            # 随记种子
            generator = torch.Generator(device=bald_model.device)
            generator.manual_seed(seed)
            comfy_pbar = ProgressBar(steps)

            def callback_bar(step, timestep, latents):
                comfy_pbar.update(1)

            with torch.no_grad():
                # 采样，变光头
                result_image = bald_model(
                    prompt="",
                    negative_prompt="",
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=W,
                    height=H,
                    image=im_tensor.unsqueeze(0),
                    controlnet_conditioning_scale=strength,
                    generator=None,
                    return_dict=False,
                    output_type="pt",
                    callback=callback_bar,
                )[0]

            # b, c, h, w -> b, h, w, c
            result_image = result_image.permute(0, 2, 3, 1)

            _images.append(result_image)

        out_images = torch.cat(_images, dim=0)

        return out_images,


class ApplyHairTransfer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HAIR_MODEL",),
                "images": ("IMAGE",),
                "bald_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "adapter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "hair/transfer"

    def apply(self, model, images, bald_image, seed, steps, cfg, control_strength, adapter_strength):
        _images = []
        _masks = []

        for image in images:
            h, w, c = image.shape

            prompt = ""
            n_prompt = ""

            # 设置adapter强度
            set_scale(model.unet, adapter_strength)
            # 随记种子
            generator = torch.Generator(device=model.device)
            generator.manual_seed(seed)
            comfy_pbar = ProgressBar(steps)

            def callback_bar(step, timestep, latents):
                comfy_pbar.update(1)

            ref_image_np = (image.cpu().numpy() * 255).astype(numpy.uint8)
            bald_image_np = (bald_image.squeeze(0).cpu().numpy() * 255).astype(numpy.uint8)
            with torch.no_grad():
                # 采样，转移发型
                result_image = model(
                    prompt,
                    negative_prompt=n_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=w,
                    height=h,
                    controlnet_condition=bald_image_np,
                    controlnet_conditioning_scale=control_strength,
                    generator=generator,
                    ref_image=ref_image_np,
                    output_type="tensor",
                    callback=callback_bar,
                    return_dict=False
                )

            # b, h, w, c
            result_image = result_image.unsqueeze(0)

            _images.append(result_image)

        out_images = torch.cat(_images, dim=0)

        return out_images,


NODE_CLASS_MAPPINGS = {
    "LoadStableHairRemoverModel": LoadStableHairRemoverModel,
    "LoadStableHairTransferModel": LoadStableHairTransferModel,
    "ApplyHairRemover": ApplyHairRemover,
    "ApplyHairTransfer": ApplyHairTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStableHairRemoverModel": "LoadStableHairRemoverModel",
    "LoadStableHairTransferModel": "LoadStableHairTransferModel",
    "ApplyHairRemover": "ApplyHairRemover",
    "ApplyHairTransfer": "ApplyHairTransfer",
}
