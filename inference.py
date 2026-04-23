from diffusers import UNet2DConditionModel
from pipeline_stable_diffusion_double_unet import StableDiffusionPipeline
import torch
import copy
from diffusers import DPMSolverMultistepScheduler
import argparse
import os
from tqdm import tqdm
import pandas as pd


class GenData:
    def __init__(self, device, pipelines, guidance_scale= 7.5, num_inference_steps= 50):
        self.pipe = pipelines
        self.pipe.safety_checker = None
        self.pipe = self.pipe.to(device)
        
        # Generating settings
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.gs = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator = torch.Generator(device= device)
        self.generator = self.generator.manual_seed(0)

        
    def gen_image(self, input_file, output_folder):
        # Make folders
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if input_file.endswith('.csv'):
            data = []
            dataset = pd.read_csv(input_file, lineterminator='\n')

            for _iter, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing images"):
                if "adv_prompt" in data:
                    target_prompt = data['adv_prompt']
                    case_num = _iter
                elif "sensitive prompt" in data:
                    target_prompt = data["sensitive prompt"]
                    case_num = _iter
                elif "prompt" in data:
                    target_prompt = data["prompt"]
                    if pd.isna(target_prompt):
                        target_prompt = ''
                    case_num = data["case_number"] if "case_number" in data else _iter
                
                guidance = data.guidance if hasattr(data,'guidance') else 7.5
                im = self.pipe( prompt = target_prompt, 
                                num_inference_steps = self.num_inference_steps,
                                guidance_scale = guidance,
                                generator = self.generator if 'evaluation_seed' not in data else torch.Generator(device= self.device).manual_seed(data["evaluation_seed"])).images[0]
                im.save(os.path.join(output_folder, f"{case_num}.png"))
            return True
        else: 
            print('Invalid input file format')
            return False

def mix_models(module1, module2, alpha=0.5):
    """
    Mix the parameters of two PyTorch modules with a given ratio.

    Args:
        module1 (nn.Module): The first PyTorch module.
        module2 (nn.Module): The second PyTorch module.
        alpha (float): The ratio of mixing, where 0 <= alpha <= 1.
                       alpha=0.5 means equal contribution from both models.

    Returns:
        nn.Module: The mixed PyTorch module (same as module1).
    """
    device = next(module1.parameters()).device
    module2 = module2.to(device)

    for param1, param2 in zip(module1.parameters(), module2.parameters()):
        param1.data = alpha * param1.data + (1 - alpha) * param2.data

    return module1


def load_bpo_pipeline(device, merge_weight=0.5, pos_path=None, neg_path=None):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    unet = UNet2DConditionModel.from_pretrained(pos_path, subfolder='unet', torch_dtype=torch.float16)
    pipe.unet = unet
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)
    pipe_tmp.unet = mix_models(
        pipe_tmp.unet, pipe.unet, merge_weight
    )
    neg_path = neg_path + '/pytorch_lora_weights.safetensors'
    lora_dir = os.path.dirname(neg_path)
    lora_file = os.path.basename(neg_path)
    pipe_tmp.load_lora_weights(lora_dir, weight_name=lora_file)
    pipe_tmp.fuse_lora()

    negative_unet = copy.deepcopy(pipe_tmp.unet)
    del pipe_tmp
    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_false"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cfg", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--pos_path", default='real-outputs/pos', type=str)
    parser.add_argument("--neg_path", default='real-outputs/neg', type=str)
    parser.add_argument("--merge_weight", default=0.0, type=float)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--save_path', type=str, required=False, default='./results/bpo/p4d')
    parser.add_argument('--prompts_path', type=str, required=False, default='./p4d.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    pipelines = load_bpo_pipeline(args.device, args.merge_weight, args.pos_path, args.neg_path)

    model = GenData(    device = args.device,
                        pipelines = pipelines,
                        guidance_scale = 7.5,
                        num_inference_steps = args.num_inference_steps)
    
    prompt_file = args.prompts_path
    save_path = args.save_path

    print(f"\n{'='*60}")
    print(f"Processing: {prompt_file}")
    print(f"Saving to: {save_path}")
    print(f"{'='*60}\n")

    model.gen_image(input_file=prompt_file, output_folder=save_path)
