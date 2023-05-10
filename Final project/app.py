from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import base64
import io
import PIL
from PIL import Image
import threading

import torch, gc
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import PIL_INTERPOLATION
from diffusers import DPMSolverMultistepScheduler

app = Flask(__name__)
CORS(app)

images = []
requests = []

def img2img(input_image, prev_image):
    # Initialize & config model
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, local_files_only=True, torch_dtype=torch.float16)

    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    # pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Some model config
    prompt = ''
    strength = 0.4
    guidance_scale = 7.5
    negative_prompt = None
    prompt_embeds = None
    negative_prompt_embeds = None
    num_inference_steps = 5
    batch_size = 1
    num_images_per_prompt = 1
    dtype = torch.float16

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(prompt=prompt, strength=strength, callback_steps=1,
                    negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = pipe._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    def preprocess(image):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image

    # 4.5 load new scheduler
    # https://huggingface.co/docs/diffusers/v0.16.0/en/stable_diffusion#memory
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 5. set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    generator = torch.manual_seed(42)
    eta = 0.0
    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    def denoise(latents):
        callback = None
        callback_steps = 1

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_max_memory_allocated()
                gc.collect()
        return latents

    # 4. Preprocess image
    image = preprocess(input_image)

    # 6. Prepare latent variables
    latent = pipe.prepare_latents(
        image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
    )

    latents = []

    if prev_image != None:
        p_image = preprocess(prev_image)
        p_latent = pipe.prepare_latents(
            p_image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )
        weights = [0.1, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        for w in weights:
            latents.append(torch.lerp(p_latent, latent, w))
    else:
        latents.append(latent)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()

    # denoised = denoise(latent)
    denoised = []
    for l in latents:
        l_d = denoise(l)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        denoised.append(l_d)

    # 9. Post-processing
    denoised_imgs = []
    with torch.no_grad():
        for l_d in denoised:
            denoised_imgs.append(pipe.decode_latents(l_d))
        # decoded = pipe.decode_latents(denoised)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

    # 11. Convert to PIL
    pil_imgs = []
    for d in denoised_imgs:
        pil_imgs.append(pipe.image_processor.postprocess(d, output_type='pil')[0])
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

    return pil_imgs

print('\nFinished model setup\n')

@app.route('/')
def hello_world():
    return 'Hello, world!'

@app.route('/send-image', methods=['POST'])
def send_image():
    data = request.get_json()
    images.append(data.get('image_str'))
    print('Received sth')

    last_image = images[-1].split(',')[1]
    if len(images) > 1:
        prev_image = images[-2].split(',')[1]
        prev_msg = base64.b64decode(prev_image)
        prev_buf = io.BytesIO(prev_msg)
        prev_img = Image.open(prev_buf)
        prev_img = prev_img.convert('RGB').resize((300, 225))
    else:
        prev_img = None
    msg = base64.b64decode(last_image)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    img = img.convert('RGB').resize((300, 225))

    output_imgs = img2img(img, prev_img)
    response = dict()
    for i, img in enumerate(output_imgs):
        data = io.BytesIO()
        img.save(data, "PNG")
        encoded_img = base64.b64encode(data.getvalue())
        decoded_img = encoded_img.decode('utf-8')
        img_data = f"data:image/jpeg;base64,{decoded_img}"
        response[f'data_{i}'] = img_data
    return response