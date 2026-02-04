
import os
import gc
import time
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torchvision
from PIL import Image
from torchvision import transforms
import argparse
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import T5EncoderModel, T5TokenizerFast
from transformers import CLIPTextModel, CLIPTokenizer

from optimum.quanto import freeze, qfloat8, quantize

from groundingdino.util.inference import Model
from functools import partial
from piplines.flux_inference_pipline import FluxPipeline
from piplines.flux_inv_pipline import Flux_inv_Pipeline
from piplines.Flux_transformer2d import FluxTransformer2DModel
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from src.flux.modules.image_embedders import ReduxImageEncoder
from src.flux.get_sim import ImageSimilarityModel

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    """Spherical linear interpolation"""
    from torch import lerp, zeros_like

    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    dot = (v0 * v1).sum(-1)
    dot_mag = dot.abs()
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    out = zeros_like(v0)

    if gotta_lerp.any():
        lerped = lerp(v0, v1, t)
        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    if can_slerp.any():
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        theta_t = theta_0 * t
        s0 = (theta_0 - theta_t).sin() / sin_theta_0
        s1 = theta_t.sin() / sin_theta_0
        slerped = s0 * v0 + s1 * v1
        out = slerped.where(can_slerp.unsqueeze(-1), out)

    return out


def performance_score(image_sim_1_3: torch.Tensor,
                      image_sim_2_3: torch.Tensor,
                      text_sim_1_3: torch.Tensor,
                      text_sim_2_3: torch.Tensor) -> float:

    norm_text_1_3 = (text_sim_1_3 - 0.15) / (0.45 - 0.15)
    norm_text_2_3 = (text_sim_2_3 - 0.15) / (0.45 - 0.15)
    score = (
        image_sim_1_3 + image_sim_2_3
        + (norm_text_2_3 + norm_text_1_3)
        - abs(image_sim_1_3 - image_sim_2_3)
        - abs(norm_text_1_3 - norm_text_2_3)
    )
    return float(score.item())


def sanitize_name(name: str) -> str:

    name = name.replace("2", " ").replace("1", "")
    return " ".join(name.split())


def build_prompts(obj1: str, obj2: str) -> Tuple[List[str], List[str], List[str], List[str]]:

    prompt = [f"A photo of {obj1} creatively fused with {obj2}."]
    prompt1 = [f"A  {obj1} ."]
    prompt2 = [f"A  {obj2} ."]
    two_prompt = [prompt[0], prompt[0]]
    return prompt, prompt1, prompt2, two_prompt



def get_xxyy(grounding_dino_model: Model,
             image: Image.Image,
             classes: List[str],
             box_threshold: float = 0.25,
             text_threshold: float = 0.25,
             nms_threshold: float = 0.8) -> Optional[Any]:

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if detections is None or len(detections.xyxy) == 0:
        return None

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        nms_threshold
    ).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    if len(detections.xyxy) == 0:
        return None
    return detections.xyxy[0]


# =========================
# 3) Scheduler / Pipeline Load
# =========================
def hacked_state_dict(self, *args, **kwargs):
    orig_state_dict = self.orig_state_dict(*args, **kwargs)
    new_state_dict = {}
    for key, value in orig_state_dict.items():
        if key.endswith("._scale"):
            continue
        if key.endswith(".input_scale"):
            continue
        if key.endswith(".output_scale"):
            continue
        if key.endswith("._data"):
            key = key[:-6]
            scale = orig_state_dict[key + "._scale"]
            # scale is the original dtype
            dtype = scale.dtype
            scale = scale.float()
            value = value.float()
            dequantized = value * scale
            
            # handle input and output scaling if they exist
            input_scale = orig_state_dict.get(key + ".input_scale")
            
            if input_scale is not None:
                # make sure the tensor is 1.0
                if input_scale.item() != 1.0:
                    raise ValueError("Input scale is not 1.0, cannot dequantize")
                
            output_scale = orig_state_dict.get(key + ".output_scale")
            
            if output_scale is not None:
                # make sure the tensor is 1.0
                if output_scale.item() != 1.0:
                    raise ValueError("Output scale is not 1.0, cannot dequantize")
            
            new_state_dict[key] = dequantized.to('cpu', dtype=dtype)
        else:
            new_state_dict[key] = value
    return new_state_dict
FLUX_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}
def patch_dequantization_on_save(model):
    model.orig_state_dict = model.state_dict
    model.state_dict = partial(hacked_state_dict, model)

def get_sampler(kwargs: dict = None) -> CustomFlowMatchEulerDiscreteScheduler:

    sched_init_args = {}
    if kwargs is not None:
        sched_init_args.update(kwargs)

    config = copy.deepcopy(FLUX_SCHEDULER_CONFIG)
    config.update(sched_init_args)
    return CustomFlowMatchEulerDiscreteScheduler.from_config(config)


def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16, 
               device0: torch.device = torch.device("cuda:0"),
            device1: Optional[torch.device] = None,
            device_mode: str = "single",) -> Tuple[FluxPipeline, Flux_inv_Pipeline]:

    with torch.no_grad():
        dev0 = device0
        dev1 = device1 if (device_mode == "split" and device1 is not None) else dev0
        noise_scheduler = get_sampler({"prediction_type": "epsilon"})

        base_model_path = model_path
        transformer_path = model_path
        subfolder = "transformer"

        if os.path.exists(transformer_path):
            subfolder = None
            transformer_path = os.path.join(transformer_path, "transformer")
            te_folder_path = os.path.join(model_path, "text_encoder")
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        # 2) Transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(dev0, dtype=dtype)
        
        # 3) VAE & scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype).to(dev0, dtype=dtype)

        # 4) T5 tokenizer + encoder_2
        tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_model_path, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(dev1, dtype=dtype)

        flush()

        # 5) CLIP text encoder
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        ).to(dev1, dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        if device_mode == "split":
            patch_dequantization_on_save(transformer)
            quantization_type = qfloat8

            quantize(transformer, weights=quantization_type)
            freeze(transformer)
            quantize(text_encoder, weights=quantization_type)
            freeze(text_encoder)
            quantize(text_encoder_2, weights=quantization_type)
            freeze(text_encoder_2)
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        inv_pipe = Flux_inv_Pipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )

        inv_pipe.text_encoder_2 = text_encoder_2
        inv_pipe.transformer = transformer.to(dev0)
        inv_pipe.scheduler = noise_scheduler

        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer.to(dev0)
        pipe.scheduler = noise_scheduler

        for enc in [pipe.text_encoder, pipe.text_encoder_2]:
            enc.requires_grad_(False)
            enc.eval()
        flush()

        return pipe, inv_pipe


@dataclass
class RunConfig:
    image_folder_path: str
    save_image_folder_path: str

    flux_ckpt_path: str ="black-forest-labs/FLUX.1-Krea-dev" #"ckpts/flux_krea"
    grounding_dino_config: str = "GroundingDino/Ground_DINO_config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_ckpt: str = "ckpts/groundingdino_swint_ogc.pth"
    efficientsam_ckpt: str = "ckpts/efficientsam_s_gpu.jit"  

    seed: int = 42
    height: int = 512
    width: int = 512
    prompt_device: str = "cuda:0"
    
    classes: Tuple[str, ...] = ("most prominent object",)

    # alpha 搜索参数
    alpha_low: float = 0.4
    alpha_high: float = 0.6
    alpha_tol: float = 0.002
    alpha_max_iter: int = 15

    # beta 搜索参数
    beta_low: float = 0.5
    beta_high: float = 3.5
    beta_tol: float = 0.5
    beta_step: float = 0.2
    beta_max_iter: int = 15

    # 分支阈值
    early_stop_score: float = 2.4

    # Redux 条件提取参数
    redux_num_steps: int = 4
    redux_guidance: float = 0.0
    redux_offload: bool = True

    # pipe/inv_pipe 推理参数
    inv_guidance: float = 5.0
    inv_steps: int = 20

    gen_guidance: float = 4.0
    gen_steps: int = 20

    max_sequence_length: int = 512
    # device
    device_mode: str = "single"   # "single" or "split"
    device0: str = "cuda:0"
    device1: str = "cuda:1"

class FusionRunner:


    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        if cfg.device_mode == "split" and torch.cuda.device_count() < 2:
            print("[WARN] device_mode=split 但检测到 GPU<2，已自动回退到 single")
            cfg.device_mode = "single"

        self.dev0 = torch.device(cfg.device0)
        self.dev1 = torch.device(cfg.device1) if cfg.device_mode == "split" else self.dev0

        # 主推理设备（Transformer/VAE/Redux）
        self.device = self.dev0
        # 文本编码设备（TextEncoder）
        self.prompt_device = self.dev1
        self.cfg.prompt_device = str(self.prompt_device)

        self.pipe, self.inv_pipe = load_model(
            cfg.flux_ckpt_path,
            dtype=torch.bfloat16,
            device0=self.dev0,
            device1=self.dev1,
            device_mode=cfg.device_mode,
        )
        self.pipe.enable_attention_slicing()
        self.inv_pipe.enable_attention_slicing()

        self.grounding_dino_model = Model(
            model_config_path=cfg.grounding_dino_config,
            model_checkpoint_path=cfg.grounding_dino_ckpt,
        )

        
        self.sim_device = self.dev1 if cfg.device_mode == "split" else self.dev0
        self.sim_model = ImageSimilarityModel(self.sim_device)

        with torch.no_grad():
            self.img_embedder = ReduxImageEncoder(self.device)
        flush()


        self._img_cache: Dict[str, Dict[str, Any]] = {}
        self._text_cache: Dict[str, torch.Tensor] = {}       # prompt_str -> feat_cpu


    def _already_done(self, obj1: str, obj2: str) -> bool:

        if not os.path.exists(self.cfg.save_image_folder_path):
            return False
        for fn in os.listdir(self.cfg.save_image_folder_path):
            if f"all_iter_final_{obj1}_{obj2}" in fn:
                if f"all_iter_final_{obj1}_{obj2}2" in fn:
                    continue
                return True
        return False

    def _save_image(self, img: Image.Image, filename: str) -> str:

        os.makedirs(self.cfg.save_image_folder_path, exist_ok=True)
        path = os.path.join(self.cfg.save_image_folder_path, filename)
        img.save(path)
        return path


    def _prepare_features(self, img1: Image.Image, img2: Image.Image,
                          prompt1: List[str], prompt2: List[str]) -> Dict[str, Any]:

        img1_bbox = get_xxyy(self.grounding_dino_model, img1, list(self.cfg.classes))
        img2_bbox = get_xxyy(self.grounding_dino_model, img2, list(self.cfg.classes))

        img1_feat = self.sim_model.get_origin_image_tensor(origin_img=img1, loaded_detections=img1_bbox)
        img2_feat = self.sim_model.get_origin_image_tensor(origin_img=img2, loaded_detections=img2_bbox)

        txt1_feat = self.sim_model.get_text_features(prompt1)
        txt2_feat = self.sim_model.get_text_features(prompt2)

        return {
            "img1_bbox": img1_bbox,
            "img2_bbox": img2_bbox,
            "img1_feat": img1_feat,
            "img2_feat": img2_feat,
            "txt1_feat": txt1_feat,
            "txt2_feat": txt2_feat,
        }


    def _get_redux_conditions(self, img_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:

        img_cond1, img_cond2 = self.pipe.cli_redux(
            width=self.cfg.width,
            height=self.cfg.height,
            seed=self.cfg.seed,
            device=self.device,
            num_steps=self.cfg.redux_num_steps,
            guidance=self.cfg.redux_guidance,
            offload=self.cfg.redux_offload,
            output_dir="output",
            img_embedder=self.img_embedder,
            img_cond_paths=img_paths,
        )
        return img_cond1, img_cond2

    def _get_or_compute_img_bbox_feat(self, img_path: str, img_pil: Image.Image):

        cache = self._img_cache.get(img_path, None)
        if cache is None:
            bbox = get_xxyy(self.grounding_dino_model, img_pil, list(self.cfg.classes))
            feat = self.sim_model.get_origin_image_tensor(origin_img=img_pil, loaded_detections=bbox)
            self._img_cache[img_path] = {
                "bbox": bbox,
                "feat_cpu": feat.detach().to("cpu"),
            }
            cache = self._img_cache[img_path]
        return cache["bbox"], cache["feat_cpu"]

    def _get_or_compute_text_feat(self, prompt_list: List[str]) -> torch.Tensor:

        key = prompt_list[0]
        feat_cpu = self._text_cache.get(key, None)
        if feat_cpu is None:
            feat = self.sim_model.get_text_features(prompt_list)
            feat_cpu = feat.detach().to("cpu")
            self._text_cache[key] = feat_cpu
        return feat_cpu
    
    def _invert_latent(self, prompt: List[str], img_cond: torch.Tensor) -> torch.Tensor:

        _, refine_latents = self.inv_pipe(
            prompt,
            height=self.cfg.height,
            width=self.cfg.width,
            guidance_scale=self.cfg.inv_guidance,
            num_inference_steps=self.cfg.inv_steps,
            max_sequence_length=self.cfg.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(self.cfg.seed),
            img_cond=img_cond,
            prompt_device=torch.device(self.cfg.prompt_device),
        )
        return refine_latents

    def _generate_and_score_two(self,
                               two_prompt: List[str],
                               latents_2: torch.Tensor,
                               img_cond_2: torch.Tensor,
                               origin_images: List[Image.Image],
                               feats: Dict[str, Any],
                               prompt1: List[str],
                               prompt2: List[str],
                               use_one_latent: bool) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:

        t0 = time.perf_counter()
        images, images_tensor = self.pipe(
            two_prompt,
            latents=latents_2,
            height=self.cfg.height,
            width=self.cfg.width,
            guidance_scale=self.cfg.gen_guidance,
            num_inference_steps=self.cfg.gen_steps,
            max_sequence_length=self.cfg.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(self.cfg.seed),
            img_cond=img_cond_2,
            origin_image=origin_images,
            use_one_lantent=use_one_latent,
            prompt_device=torch.device(self.cfg.prompt_device),
        )
        t1 = time.perf_counter()


        results = []
        for k in range(2):
            img_k = images[k].resize((self.cfg.width, self.cfg.height), resample=Image.Resampling.BILINEAR)
            bbox_k = get_xxyy(self.grounding_dino_model, img_k, list(self.cfg.classes))

            # 与 source1 的相似度
            img_1_3_sim, txt_1_3_sim = self.sim_model.get_image_tensor_image_sim(
                feats["img1_feat"],
                images_tensor[k].unsqueeze(0).to(self.sim_device, non_blocking=True),
                feats["txt1_feat"],
                loaded_detections=bbox_k,
            )
            # 与 source2 的相似度
            img_2_3_sim, txt_2_3_sim = self.sim_model.get_image_tensor_image_sim(
                feats["img2_feat"],
                images_tensor[k].unsqueeze(0).to(self.sim_device, non_blocking=True),
                feats["txt2_feat"],
                loaded_detections=bbox_k,
            )

            score = performance_score(img_1_3_sim, img_2_3_sim, txt_1_3_sim, txt_2_3_sim)
            results.append({
                "score": score,
                "img_1_3_sim": img_1_3_sim,
                "img_2_3_sim": img_2_3_sim,
                "txt_1_3_sim": txt_1_3_sim,
                "txt_2_3_sim": txt_2_3_sim,
                "bbox": bbox_k,
            })

        return images, results

    # ---------- Stage A：alpha 搜索模块（golden section, batch=2） ----------

    def search_alpha(self,
                     prompt: List[str],
                     two_prompt: List[str],
                     img_cond1: torch.Tensor,
                     img_cond2: torch.Tensor,
                     refine_latents: torch.Tensor,
                     origin_images: List[Image.Image],
                     feats: Dict[str, Any],
                     prompt1: List[str],
                     prompt2: List[str]) -> Dict[str, Any]:

        low, high = self.cfg.alpha_low, self.cfg.alpha_high
        tol = self.cfg.alpha_tol
        phi = (1 + 5 ** 0.5) / 2

        best = {
            "score": -1e9,
            "alpha": None,
            "image": None,
            "metrics": None,
        }

        it = 0
        use_one_latent = True

        while (high - low) > tol and it < self.cfg.alpha_max_iter:
            it += 1
            mid1 = round(low + (high - low) / phi, 5)
            mid2 = round(high - (high - low) / phi, 5)

            cond_mid1 = slerp(img_cond1, img_cond2, mid1)
            cond_mid2 = slerp(img_cond1, img_cond2, mid2)

            latents_2 = torch.cat((refine_latents, refine_latents), dim=0)
            cond_2 = torch.cat((cond_mid1, cond_mid2), dim=0).to("cuda")

            images, res = self._generate_and_score_two(
                two_prompt=two_prompt,
                latents_2=latents_2,
                img_cond_2=cond_2,
                origin_images=origin_images,
                feats=feats,
                prompt1=prompt1,
                prompt2=prompt2,
                use_one_latent=use_one_latent,
            )

            score1, score2 = res[0]["score"], res[1]["score"]

            if score1 >= score2:
                low = mid2
                if score1 > best["score"]:
                    best.update({"score": score1, "alpha": mid1, "image": images[0], "metrics": res[0]})
            else:
                high = mid1
                if score2 > best["score"]:
                    best.update({"score": score2, "alpha": mid2, "image": images[1], "metrics": res[1]})

        return best


    def search_beta(self,
                    prompt: List[str],
                    two_prompt: List[str],
                    img_cond1: torch.Tensor,
                    img_cond2: torch.Tensor,
                    best_alpha: float,
                    fixed_1: bool,
                    origin_images: List[Image.Image],
                    feats: Dict[str, Any],
                    prompt1: List[str],
                    prompt2: List[str]) -> Dict[str, Any]:

        low, high = self.cfg.beta_low, self.cfg.beta_high
        tol = self.cfg.beta_tol
        step = self.cfg.beta_step
        phi = (1 + 5 ** 0.5) / 2

        true_img_cond = slerp(img_cond1, img_cond2, best_alpha)
        gen_cond_2 = torch.cat([true_img_cond, true_img_cond], dim=0).to("cuda")

        best = {
            "score": -1e9,
            "beta": None,
            "image": None,
            "metrics": None,
        }

        evaluated = set()
        it = 0
        use_one_latent = False

        while (high - low) > tol and it < self.cfg.beta_max_iter:
            it += 1

            beta_values = torch.arange(max(low, self.cfg.beta_low),
                                       min(high + step / 2, self.cfg.beta_high),
                                       step).tolist()
            beta_values = [b for b in beta_values if b not in evaluated]
            if len(beta_values) == 0:
                break

            mid1_ideal = low + (high - low) / phi
            mid2_ideal = high - (high - low) / phi

            beta_values.sort()
            if len(beta_values) == 1:
                mid1 = mid2 = beta_values[0]
            else:
                mid1 = min(beta_values, key=lambda x: abs(x - mid1_ideal))
                beta_values.remove(mid1)
                mid2 = min(beta_values, key=lambda x: abs(x - mid2_ideal))

            evaluated.add(mid1)
            evaluated.add(mid2)


            if fixed_1:
                inv_cond_mid1 = torch.cat((img_cond1 * 0.5, img_cond2 * mid1), dim=1).to("cuda")
                inv_cond_mid2 = torch.cat((img_cond1 * 0.5, img_cond2 * mid2), dim=1).to("cuda")
            else:
                inv_cond_mid1 = torch.cat((img_cond1 * mid1, img_cond2 * 0.5), dim=1).to("cuda")
                inv_cond_mid2 = torch.cat((img_cond1 * mid2, img_cond2 * 0.5), dim=1).to("cuda")

            refine1 = self._invert_latent(prompt, inv_cond_mid1)
            refine2 = self._invert_latent(prompt, inv_cond_mid2)
            latents_2 = torch.cat([refine1, refine2], dim=0)

            images, res = self._generate_and_score_two(
                two_prompt=two_prompt,
                latents_2=latents_2,
                img_cond_2=gen_cond_2,
                origin_images=origin_images,
                feats=feats,
                prompt1=prompt1,
                prompt2=prompt2,
                use_one_latent=use_one_latent,
            )

            score1, score2 = res[0]["score"], res[1]["score"]

            if score1 >= score2:
                low = mid2
                if score1 > best["score"]:
                    best.update({"score": score1, "beta": mid1, "image": images[0], "metrics": res[0]})
            else:
                high = mid1
                if score2 > best["score"]:
                    best.update({"score": score2, "beta": mid2, "image": images[1], "metrics": res[1]})

        return best


    def run_one_pair(self, img_path1: str, img_path2: str) -> None:

        obj1 = os.path.splitext(os.path.basename(img_path1))[0]
        obj2 = os.path.splitext(os.path.basename(img_path2))[0]

        if self._already_done(obj1, obj2):
            print(f"skip {obj1}_{obj2}")
            return

        print(f"origin_image_name:{obj1}")
        print(f"fuse_image_name:{obj2}")


        img1 = Image.open(img_path1).resize((self.cfg.width, self.cfg.height), resample=Image.Resampling.NEAREST)
        img2 = Image.open(img_path2).resize((self.cfg.width, self.cfg.height), resample=Image.Resampling.NEAREST)


        p_obj1 = sanitize_name(obj1)
        p_obj2 = sanitize_name(obj2)
        prompt, prompt1, prompt2, two_prompt = build_prompts(p_obj1, p_obj2)

        img1_bbox, img1_feat_cpu = self._get_or_compute_img_bbox_feat(img_path1, img1)
        img2_bbox, img2_feat_cpu = self._get_or_compute_img_bbox_feat(img_path2, img2)

        txt1_feat_cpu = self._get_or_compute_text_feat(prompt1)
        txt2_feat_cpu = self._get_or_compute_text_feat(prompt2)

        feats = {
            "img1_bbox": img1_bbox,
            "img2_bbox": img2_bbox,
            "img1_feat": img1_feat_cpu.to(self.sim_device, non_blocking=True),
            "img2_feat": img2_feat_cpu.to(self.sim_device, non_blocking=True),
            "txt1_feat": txt1_feat_cpu.to(self.sim_device, non_blocking=True),
            "txt2_feat": txt2_feat_cpu.to(self.sim_device, non_blocking=True),
        }
        flush()

        img_cond1, img_cond2 = self._get_redux_conditions([img_path1, img_path2])
        flush()


        cat_img_cond = torch.cat((img_cond1, img_cond2), dim=1).to("cuda")
        refine_latents = self._invert_latent(prompt, cat_img_cond)
        flush()


        best_a = self.search_alpha(
            prompt=prompt,
            two_prompt=two_prompt,
            img_cond1=img_cond1,
            img_cond2=img_cond2,
            refine_latents=refine_latents,
            origin_images=[img1, img2],
            feats=feats,
            prompt1=prompt1,
            prompt2=prompt2,
        )

        m = best_a["metrics"]
        a_score = round(best_a["score"], 3)
        a_alpha = best_a["alpha"]
        fn_a = (
            f"best_golden_{obj1}_{obj2}_current_score_{a_score}"
            f"_TI_1_3_sim_{round(float(m['txt_1_3_sim']),3)}_{round(float(m['img_1_3_sim'].item()),3)}"
            f"_TI2_3_sim_{round(float(m['txt_2_3_sim']),3)}_{round(float(m['img_2_3_sim'].item()),3)}"
            f"_current_alpha_{a_alpha}.png"
        )
        self._save_image(best_a["image"], fn_a)
        flush()

        final_best_image = best_a["image"]
        final_best_score = best_a["score"]
        final_best_alpha = best_a["alpha"]
        final_best_beta = 0.0
        final_fixed1 = 0

        
        if best_a["score"] > self.cfg.early_stop_score:
            fn_final = (
                f"all_iter_final_{obj1}_{obj2}_current_score_{round(final_best_score,3)}"
                f"_TI_1_3_sim_{round(float(m['txt_1_3_sim']),3)}_{round(float(m['img_1_3_sim'].item()),3)}"
                f"_TI2_3_sim_{round(float(m['txt_2_3_sim']),3)}_{round(float(m['img_2_3_sim'].item()),3)}"
                f"_cbeta_{final_best_beta}_calpha_{final_best_alpha}_fix1_{final_fixed1}.png"
            )
            self._save_image(final_best_image, fn_final)
            return

        if (m["img_1_3_sim"] + (m["txt_1_3_sim"] *3.13)) > (m["img_2_3_sim"] + (m["txt_2_3_sim"] *3.13)):
            fixed_1 = True
        else:
            fixed_1 = False

        best_b = self.search_beta(
            prompt=prompt,
            two_prompt=two_prompt,
            img_cond1=img_cond1,
            img_cond2=img_cond2,
            best_alpha=final_best_alpha,
            fixed_1=fixed_1,
            origin_images=[img1, img2],
            feats=feats,
            prompt1=prompt1,
            prompt2=prompt2,
        )

        mb = best_b["metrics"]
        b_score = round(best_b["score"], 3)
        b_beta = best_b["beta"]
        fn_b = (
            f"next_best_golden_{obj1}_{obj2}_current_score_{b_score}"
            f"_TI_1_3_sim_{round(float(mb['txt_1_3_sim']),3)}_{round(float(mb['img_1_3_sim'].item()),3)}"
            f"_TI2_3_sim_{round(float(mb['txt_2_3_sim']),3)}_{round(float(mb['img_2_3_sim'].item()),3)}"
            f"_current_beta_{b_beta}_fix1_{int(fixed_1)}.png"
        )
        self._save_image(best_b["image"], fn_b)
        flush()

        if best_b["score"] > final_best_score:
            final_best_score = best_b["score"]
            final_best_image = best_b["image"]
            final_best_beta = best_b["beta"]
            final_fixed1 = int(fixed_1)
            fn_b2 = (
                f"next_final_best_golden_{obj1}_{obj2}_current_score_{round(final_best_score,3)}"
                f"_TI_1_3_sim_{round(float(mb['txt_1_3_sim']),3)}_{round(float(mb['img_1_3_sim'].item()),3)}"
                f"_TI2_3_sim_{round(float(mb['txt_2_3_sim']),3)}_{round(float(mb['img_2_3_sim'].item()),3)}"
                f"_current_beta_{final_best_beta}_fix1_{final_fixed1}.png"
            )
            self._save_image(final_best_image, fn_b2)
            flush()


        mfinal = mb if (best_b["score"] >= best_a["score"]) else m
        fn_final = (
            f"all_iter_final_{obj1}_{obj2}_current_score_{round(final_best_score,3)}"
            f"_TI_1_3_sim_{round(float(mfinal['txt_1_3_sim']),3)}_{round(float(mfinal['img_1_3_sim'].item()),3)}"
            f"_TI2_3_sim_{round(float(mfinal['txt_2_3_sim']),3)}_{round(float(mfinal['img_2_3_sim'].item()),3)}"
            f"_cbeta_{final_best_beta}_calpha_{final_best_alpha}_fix1_{final_fixed1}.png"
        )
        self._save_image(final_best_image, fn_final)
        flush()


    def run_dataset(self) -> None:

        img_files = [
            f for f in os.listdir(self.cfg.image_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if len(img_files) < 2:
            raise ValueError("Folder must contain at least two image files.")

        for i in range(len(img_files)):
            i=len(img_files)-i-1
            for j in range(len(img_files)):
                if j == i:
                    continue
                p1 = os.path.join(self.cfg.image_folder_path, img_files[i])
                p2 = os.path.join(self.cfg.image_folder_path, img_files[j])
                self.run_one_pair(p1, p2)
def build_args() -> argparse.Namespace:
    """
    Args 
    - input_dir: origin image floder
    - output_dir: save image floder
    - seed: random seed
    - width
    """
    parser = argparse.ArgumentParser(description="Flux fusion runner args")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/opt/data/private/xzr/VMDIFF/final_dataset2",
        help="origin image floder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/data/private/xzr/VMDIFF/VMDiff_code/final_test2",
        help="save image floder"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="output width(height == width)"
    )
    parser.add_argument(
        "--device_mode",
        type=str,
        default="single",
        choices=["single", "split"],
        help="single: 全部放一张卡; split: 放两张卡"
    )
    parser.add_argument("--device0", type=str, default="cuda:0", help="主卡(Transformer/VAE/Redux)")
    parser.add_argument("--device1", type=str, default="cuda:1", help="副卡(TextEncoder/SimModel)")

    # parser.add_argument("--height", type=int, default=None, help="输出高度（默认不填则等于 width）")

    return parser.parse_args()

def main():
    args = build_args()
    height = args.width
    cfg = RunConfig(
        image_folder_path=args.input_dir,
        save_image_folder_path=args.output_dir,
        seed=args.seed,
        width=args.width,
        height=height,
        device_mode=args.device_mode,
        device0=args.device0,
        device1=args.device1,
        
    )

    runner = FusionRunner(cfg)
    runner.run_dataset()


if __name__ == "__main__":
    main()
