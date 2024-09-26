#!/usr/bin/env python3
import os
import requests
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import multiprocessing
import numpy as np
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from dotenv import load_dotenv

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from flask import (
    Flask,
    request,
    send_file,
    make_response,
)
from flask_socketio import SocketIO

# Disable ability for Flask to display warning about using a development server in a production environment.
# https://gist.github.com/jerblack/735b9953ba1ab6234abb43174210d356
from flask_cors import CORS

from lama_cleaner.helper import (
    load_img,
    resize_max_size,
    pil_to_bytes,
)


NUM_THREADS = str(multiprocessing.cpu_count())

# fix libomp problem on windows https://github.com/Sanster/lama-cleaner/issues/56
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]
BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app, expose_headers=["Content-Disposition"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
model: ModelManager = None
output_dir: str = None
device = None
input_image_path: str = None
is_disable_model_switch: bool = False
is_controlnet: bool = False
controlnet_method: str = "control_v11p_sd15_canny"
is_enable_file_manager: bool = False
image_quality: int = 95
plugins = {}

def RemoveCudaMemory():
    try:
        import gc
        gc.collect()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        torch.cuda.empty_cache()
    except:
        pass
    return
def get_image_extension(orignal_image):
    # Extract the file extension
    _, ext = os.path.splitext(orignal_image)
    # Remove the dot from the extension and convert to lowercase
    file_extension = ext.lstrip('.').lower()
    if file_extension == 'jpg' or file_extension == 'jpeg':
        file_extension = 'jpeg'
    return file_extension


def diffuser_callback(i, t, latents):
    socketio.emit("diffusion_progress", {"step": i})

def main(args):
    global model
    global device
    global input_image_path
    global is_disable_model_switch
    global is_enable_file_manager
    global thumb
    global output_dir
    global is_controlnet
    global controlnet_method
    global image_quality


    image_quality = args.quality

    output_dir = args.output_dir

    device = torch.device(args.device)
    is_disable_model_switch = args.disable_model_switch


    input_image_path = args.input

    model = ModelManager(
        name=args.model,
        sd_controlnet=args.sd_controlnet,
        device=device,
        no_half=args.no_half,
        hf_access_token=args.hf_access_token,
        disable_nsfw=args.sd_disable_nsfw or args.disable_nsfw,
        sd_cpu_textencoder=args.sd_cpu_textencoder,
        sd_run_local=args.sd_run_local,
        local_files_only=args.local_files_only,
        cpu_offload=args.cpu_offload,
        enable_xformers=args.sd_enable_xformers or args.enable_xformers,
        callback=diffuser_callback,
    )

    if args.gui:
        app_width, app_height = args.gui_size
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app,
            socketio=socketio,
            width=app_width,
            height=app_height,
            host=args.host,
            port=args.port,
            close_server_on_exit=not args.no_gui_auto_close,
        )
        ui.run()
    else:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True,
        )

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app.route("/RemoveObjects", methods=["POST"])
def ProcessRemoveObjects():
    try:
        import io
        import random
        import cv2
        from PIL import Image
        RemoveCudaMemory()

        form = request.form
        image_url = form["image"]
        mask_url = form["mask"]
        image_response = requests.get(image_url)
        mask_response = requests.get(mask_url)
        origin_image_bytes = image_response.content
        maskImage = mask_response.content


        ext = get_image_extension(image_url)
        # RGB
        image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)
        mask, _ = load_img(maskImage, gray=True)
        mask = cv2.threshold(mask, 243, 255, cv2.THRESH_BINARY)[1]

        if image.shape[:2] != mask.shape[:2]:
            return (
                f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
                400,
            )

        interpolation = cv2.INTER_CUBIC

        size_limit = max(image.shape)
        paint_by_example_example_image = None
        config = Config(
            ldm_steps=int(form["ldmSteps"]),
            ldm_sampler=form["ldmSampler"],
            hd_strategy=form["hdStrategy"],
            zits_wireframe=form["zitsWireframe"],
            hd_strategy_crop_margin=form["hdStrategyCropMargin"],
            hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
            hd_strategy_resize_limit=int(form["hdStrategyResizeLimit"]),
            prompt=form["prompt"],
            negative_prompt=form["negativePrompt"],
            use_croper=form["useCroper"],
            croper_x=form["croperX"],
            croper_y=form["croperY"],
            croper_height=form["croperHeight"],
            croper_width=form["croperWidth"],
            sd_scale=float(form["sdScale"]),
            sd_mask_blur=float(form["sdMaskBlur"]),
            sd_strength=float(form["sdStrength"]),
            sd_steps=float(form["sdSteps"]),
            sd_guidance_scale=float(form["sdGuidanceScale"]),
            sd_sampler=form["sdSampler"],
            sd_seed=float(form["sdSeed"]),
            sd_match_histograms=form["sdMatchHistograms"],
            cv2_flag=form["cv2Flag"],
            cv2_radius=form["cv2Radius"],
            paint_by_example_steps=form["paintByExampleSteps"],
            paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
            paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
            paint_by_example_seed=form["paintByExampleSeed"],
            paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
            paint_by_example_example_image=paint_by_example_example_image,
            p2p_steps=form["p2pSteps"],
            p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
            p2p_guidance_scale=form["p2pGuidanceScale"],
            controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
            controlnet_method=form["controlnet_method"],
        )

        if config.sd_seed == -1:
            config.sd_seed = random.randint(1, 999999999)
        if config.paint_by_example_seed == -1:
            config.paint_by_example_seed = random.randint(1, 999999999)

        image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
        mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

        try:
            res_np_img = model(image, mask, config)
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                # NOTE: the string may change?
                return "CUDA out of memory", 500
            else:
                return f"{str(e)}", 500
        finally:
            torch_gc()

        res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
                )
            res_np_img = np.concatenate(
                (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
            )

        bytes_io = io.BytesIO(
            pil_to_bytes(
                Image.fromarray(res_np_img),
                ext,
                quality=image_quality,
                exif_infos=exif_infos,
            )
        )
        response = make_response(
            send_file(
                bytes_io,
                mimetype=f"image/{ext}",
            )
        )
        response.headers["X-Seed"] = str(config.sd_seed)
        socketio.emit("diffusion_finish")
        RemoveCudaMemory()
        return response
    except Exception as e:
        return make_response(str(e), 403)

load_dotenv()

