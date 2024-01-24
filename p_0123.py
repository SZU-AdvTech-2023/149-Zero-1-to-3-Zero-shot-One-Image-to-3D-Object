# A diffuser version implementation of Zero1to3 (https://github.com/cvlab-columbia/zero123), ICCV 2023
# by Xin Kong

import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import einops
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import FrozenDict
import PIL
import numpy as np
import math
import kornia
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# todo
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class CCProjection(ModelMixin, ConfigMixin):
    def __init__(self, a=772, b=768):
        super().__init__()
        self.a = a
        self.b = b
        self.projection = torch.nn.Linear(a, b)

    def forward(self, x):
        # 执行投影操作
        return self.projection(x)


class p_0123(DiffusionPipeline):
    r"""
    Zero1to3 使用的单视图条件生成新视图的流水线。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档，了解库为所有流水线实现的通用方法（例如下载或保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示。
        image_encoder ([`CLIPVisionModelWithProjection`]):
            冻结的 CLIP 图像编码器。Stable Diffusion Image Variation 使用 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection) 的视觉部分，具体而言是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变种。
        tokenizer (`CLIPTokenizer`):
            类 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的分词器。
        unet ([`UNet2DConditionModel`]): 
            有条件 U-Net 架构，用于去噪编码的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度程序，用于去噪编码的图像潜在表示。可以是 [`DDIMScheduler`]、[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            估计生成的图像是否可能被视为具有冒犯性或有害的分类模块。
            有关详细信息，请参阅[模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`CLIPFeatureExtractor`]):
            从生成的图像中提取特征的模型，用作 `safety_checker` 的输入。
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        cc_projection: CCProjection,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"此调度程序的配置文件已过时：{scheduler}。`steps_offset` 应该设置为 1，而不是 {scheduler.config.steps_offset}。"
                f"请确保相应地更新配置，因为保留 `steps_offset` 可能导致将来版本中的错误结果。"
                "如果您从 Hugging Face Hub 下载了此检查点，如果您能为 `scheduler/scheduler_config.json` 文件提交一个 Pull 请求，将会非常感谢。"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"此调度程序的配置文件已设置：{scheduler} 未设置配置 `clip_sample`。"
                "在配置文件中，`clip_sample` 应设置为 False。请确保相应地更新配置，因为在配置中不设置 `clip_sample` 可能导致将来版本中的错误结果。"
                "如果您从 Hugging Face Hub 下载了此检查点，如果您能为 `scheduler/scheduler_config.json` 文件提交一个 Pull 请求，将会非常感谢。"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"您通过传递 `safety_checker=None` 来禁用了 {self.__class__} 的安全检查器。请确保您遵守 Stable Diffusion 许可协议的条件，不要在向公众开放的服务或应用程序中公开未经筛选的结果。"
                "Diffusers 团队和 Hugging Face 强烈建议在所有面向公众的情况下保持安全过滤器启用，仅在涉及分析网络行为或审计其结果的用例中禁用它。"
                "有关更多信息，请参阅 https://github.com/huggingface/diffusers/pull/254 。"
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                f"在加载 {self.__class__} 时，请确保定义一个特征提取器，以便使用安全检查器。"
                "如果不想使用安全检查器，可以传递 `'safety_checker=None'`。"
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "UNet 的配置文件将默认的 `sample_size` 设置为小于 64，这似乎极不可能。"
                "如果您的检查点是以下任一版本的精细调整版本：\n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n 您应该在配置文件中将 'sample_size' 更改为 64。"
                "请确保相应地更新配置，因为在配置中保留 `sample_size=32` 可能导致将来版本中的错误结果。"
                "如果您从 Hugging Face Hub 下载了此检查点，如果您能为 `unet/config.json` 文件提交一个 Pull 请求，将会非常感谢。"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            cc_projection=cc_projection,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # self.model_mode = None

    def enable_vae_slicing(self):
        r"""
        启用切片 VAE 解码。

        当启用此选项时，VAE 将在几个步骤中对输入张量进行切片以进行解码。这对于节省一些内存并允许更大的批量大小非常有用。
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        禁用切片 VAE 解码。如果先前调用了 `enable_vae_slicing`，则此方法将恢复在一个步骤中进行解码。
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        启用平铺 VAE 解码。

        当启用此选项时，VAE 将在几个步骤中将输入张量分割成瓦片以进行解码和编码。这对于节省大量内存并允许处理更大的图像非常有用。
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        禁用平铺 VAE 解码。如果先前调用了 `enable_vae_tiling`，则此方法将恢复在一个步骤中进行解码。
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        使用 accelerate 将所有模型卸载到 CPU，显著减少内存使用。调用此方法时，unet、text_encoder、vae 和 safety checker 的状态字典将保存到 CPU，
        然后移动到 `torch.device('meta')`，仅在它们的特定子模块调用其 `forward` 方法时才加载到 GPU。
        请注意，卸载是基于子模块的。与 `enable_model_cpu_offload` 相比，内存节省更多，但性能较低。
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def C_p(self, x):
        dtype = x.dtype
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                     torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return x

    def ep(self, image, pose, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        # 预处理图片
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        image = image.to(device=device, dtype=dtype)
        image = self.C_p(image)
        img_prompt_embeds = self.image_encoder(image).image_embeds.to(dtype=dtype)
        img_prompt_embeds = img_prompt_embeds.unsqueeze(1)

        bs_embed, seq_len, _ = img_prompt_embeds.shape
        img_prompt_embeds = img_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        img_prompt_embeds = img_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        
        dtype = next(self.cc_projection.parameters()).dtype
        if isinstance(pose, torch.Tensor):
            pose_prompt_embeds = pose.unsqueeze(1).to(device=device, dtype=dtype)
        else:
            if isinstance(pose[0], list):
                pose = torch.Tensor(pose)
            else:
                pose = torch.Tensor([pose])
            x, y, z = pose[:,0].unsqueeze(1), pose[:,1].unsqueeze(1), pose[:,2].unsqueeze(1)
            pose_prompt_embeds = torch.cat([torch.deg2rad(x),
                                         torch.sin(torch.deg2rad(y)),
                                         torch.cos(torch.deg2rad(y)),
                                         z], dim=-1).unsqueeze(1).to(device=device, dtype=dtype)  # B, 1, 4
        # duplicate pose embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = pose_prompt_embeds.shape
        pose_prompt_embeds = pose_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pose_prompt_embeds = pose_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
        prompt_embeds = self.cc_projection(prompt_embeds)
        if do_classifier_free_guidance:
            negative_prompt = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"您传递了长度为 {len(generator)} 的生成器列表，但请求的有效批量大小为 {batch_size}。确保批量大小与生成器列表的长度匹配。")

        latents = latents.to(device) if latents is not None else randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents *= self.scheduler.init_noise_sigma
        return latents

    def prepare_img_latents(self, image, batch_size, dtype, device, generator=None, do_classifier_free_guidance=False):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError("`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.")

        if isinstance(generator, list):
            init_latents = [self.vae.encode(image[i:i+1]).latent_dist.mode(generator[i]) for i in range(batch_size)]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > init_latents.shape[0]:
            num_images_per_prompt = batch_size // init_latents.shape[0]
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1).repeat(1, num_images_per_prompt, 1, 1, 1).view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        init_latents = torch.cat([torch.zeros_like(init_latents), init_latents]) if do_classifier_free_guidance else init_latents

        init_latents = init_latents.to(device=device, dtype=dtype)
        return init_latents

    @torch.no_grad()
    def __call__(
        self,
        input_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        poses: Union[List[float], List[List[float]]] = None,
        torch_dtype=torch.float32,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):

        # 0. 定义长和宽
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. 检查输入
        self.check_inputs(input_imgs, height, width, callback_steps)

        # 2. 定义所有参数
        batch_size = 1 if isinstance(input_imgs, PIL.Image.Image) else len(input_imgs) if isinstance(input_imgs, list) else input_imgs.shape[0]
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. 编码输入图像和姿态
        prompt_embeds = self.ep(prompt_imgs, poses, device, num_images_per_prompt, do_classifier_free_guidance)

        # 4. 时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. 准备latent变量
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. 准备图片latents
        img_latents = self.prepare_img_latents(input_imgs,
                                               batch_size * num_images_per_prompt,
                                               prompt_embeds.dtype,
                                               device,
                                               generator,
                                               do_classifier_free_guidance)

        # 7. 准备额外 step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. 去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, img_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred.to(dtype=torch.float32), t, latents.to(dtype=torch.float32)).prev_sample.to(prompt_embeds.dtype)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        has_nsfw_concept = None
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)