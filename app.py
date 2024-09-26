import base64
import io
import random
import cv2
from PIL import Image
import requests
import numpy as np
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from lama_cleaner.helper import (
    load_img,
    resize_max_size,
    pil_to_bytes,
)


class InferlessPythonModel:
  def initialize(self):
      self.model = ModelManager(
        name="lama",
        # sd_controlnet=args.sd_controlnet,
        device="cuda",
        # no_half=args.no_half,
        # hf_access_token=args.hf_access_token,
        # disable_nsfw=args.sd_disable_nsfw or args.disable_nsfw,
        # sd_cpu_textencoder=args.sd_cpu_textencoder,
        # sd_run_local=args.sd_run_local,
        # local_files_only=args.local_files_only,
        # cpu_offload=args.cpu_offload,
        # enable_xformers=args.sd_enable_xformers or args.enable_xformers,
        # callback=diffuser_callback,
    )

  def infer(self, inputs):
    image_url = inputs["image"]
    mask_url = inputs["mask"]
    prompt=inputs["prompt"]
    
    image_response = requests.get(image_url)
    mask_response = requests.get(mask_url)
    origin_image_bytes = image_response.content
    maskImage = mask_response.content
    
    
    # ext = get_image_extension(image_url)
    # RGB
    image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)
    mask, _ = load_img(maskImage, gray=True)
    mask = cv2.threshold(mask, 243, 255, cv2.THRESH_BINARY)[1]
    
    if image.shape[:2] != mask.shape[:2]:
         return {"error":
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]} 400"
      }
    
    interpolation = cv2.INTER_CUBIC
    
    size_limit = max(image.shape)
    paint_by_example_example_image = None

    config = Config(
        ldm_steps=int(inputs.get("ldmSteps", 50)),
        ldm_sampler=inputs.get("ldmSampler", "plms"),
        hd_strategy=inputs.get("hdStrategy", "Resize"),
        zits_wireframe=inputs.get("zitsWireframe", True),
        hd_strategy_crop_margin=inputs.get("hdStrategyCropMargin", 32),
        hd_strategy_crop_trigger_size=inputs.get("hdStrategyCropTrigerSize", 2048),
        hd_strategy_resize_limit=int(inputs.get("hdStrategyResizeLimit", 2048)),
        prompt=prompt,
        negative_prompt=inputs.get("negativePrompt", "blurry, low quality"),
        use_croper=inputs.get("useCroper", False),
        croper_x=inputs.get("croperX", 0),
        croper_y=inputs.get("croperY", 0),
        croper_height=inputs.get("croperHeight", 512),
        croper_width=inputs.get("croperWidth", 512),
        sd_scale=float(inputs.get("sdScale", 1.0)),
        sd_mask_blur=float(inputs.get("sdMaskBlur", 4)),
        sd_strength=float(inputs.get("sdStrength", 0.75)),
        sd_steps=float(inputs.get("sdSteps", 50)),
        sd_guidance_scale=float(inputs.get("sdGuidanceScale", 7.5)),
        sd_sampler=inputs.get("sdSampler", "uni_pc"),
        sd_seed=float(inputs.get("sdSeed", 42)),
        sd_match_histograms=inputs.get("sdMatchHistograms", False),
        cv2_flag=inputs.get("cv2Flag", "INPAINT_NS"),
        cv2_radius=inputs.get("cv2Radius", 4),
        paint_by_example_steps=inputs.get("paintByExampleSteps", 50),
        paint_by_example_guidance_scale=inputs.get("paintByExampleGuidanceScale", 7.5),
        paint_by_example_mask_blur=inputs.get("paintByExampleMaskBlur", 0),
        paint_by_example_seed=inputs.get("paintByExampleSeed", 42),
        paint_by_example_match_histograms=inputs.get("paintByExampleMatchHistograms", False),
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_steps=inputs.get("p2pSteps", 50),
        p2p_image_guidance_scale=inputs.get("p2pImageGuidanceScale", 1.5),
        p2p_guidance_scale=inputs.get("p2pGuidanceScale", 7.5),
        controlnet_conditioning_scale=inputs.get("controlnet_conditioning_scale", 0.4),
        controlnet_method=inputs.get("controlnet_method", "control_v11p_sd15_canny"),
    )
    
    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)
    
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    
    res_np_img = self.model(image, mask, config)    
    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )
    
    buff = io.BytesIO()
    Image.fromarray(res_np_img).save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    
    return { "generated_image_base64" : img_str }

  def finalize(self):
      pass
