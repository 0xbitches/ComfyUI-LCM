from .lcm.lcm_scheduler import LCMScheduler
from .lcm.lcm_pipeline import LatentConsistencyModelPipeline
from .lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
from os import path
import time
import torch
import random
import numpy as np
from comfy.model_management import get_torch_device

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


class LCM_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "positive_prompt": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, cfg, positive_prompt, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt=positive_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="np",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)



class LCM_img2img_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "images": ("IMAGE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompt_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "positive_prompt": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, prompt_strength, cfg, images, positive_prompt, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                custom_revision="main",
                revision="fb9c5d",
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        images = np.transpose(images, (0, 3, 1, 2))
        results = []
        for i in range(images.shape[0]):
            image = images[i]
            result = self.pipe(
                image=image,
                prompt=positive_prompt,
                strength=prompt_strength,
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                lcm_origin_steps=50,
                output_type="np",
                ).images
            tensor_results = [torch.from_numpy(np_result) for np_result in result]
            results.extend(tensor_results)

        results = torch.stack(results)
        
        print("LCM img2img inference time: ", time.time() - start_time, "seconds")

        return (results,)


class LCM_Sampler_Advanced:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "conditioning": ("CONDITIONING",),
                }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, cfg, conditioning, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt_embeds=conditioning[0][0],
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="latent",
        ).images

        print("LCM Advanced inference time: ", time.time() - start_time, "seconds")

        # scale latents with vae factor
        return ({"samples": result / self.pipe.vae.config.scaling_factor},)



class LCM_img2img_Sampler_Advanced:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "images": ("IMAGE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompt_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "conditioning": ("CONDITIONING",),
                }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, prompt_strength, cfg, images, conditioning, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        images = np.transpose(images, (0, 3, 1, 2))
        results = []
        for i in range(images.shape[0]):
            image = images[i]
            result = self.pipe(
                image=image,
                prompt_embeds=conditioning[0][0],
                strength=prompt_strength,
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                lcm_origin_steps=50,
                output_type="latent",
                ).images
            
            # scale latents with vae factor
            result = result / self.pipe.vae.config.scaling_factor
            results.extend(result)
            
        
        print("LCM img2img Advanced inference time: ", time.time() - start_time, "seconds")
    
        return ({"samples": torch.stack(results)},)




NODE_CLASS_MAPPINGS = {
    "LCM_Sampler": LCM_Sampler,
    "LCM_img2img_Sampler": LCM_img2img_Sampler,
    "LCM_Sampler_Advanced": LCM_Sampler_Advanced,
    "LCM_img2img_Sampler_Advanced": LCM_img2img_Sampler_Advanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCM_Sampler": "LCM Sampler",
    "LCM_img2img_Sampler": "LCM img2img Sampler",
    "LCM_Sampler_Advanced": "LCM Sampler (Advanced)",
    "LCM_img2img_Sampler_Advanced": "LCM img2img Sampler (Advanced)"
}
