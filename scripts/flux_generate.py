import torch
from diffusers import FluxPipeline
import datetime

def generate_selfie(prompt, filename):
    print(f"Loading FLUX.1 [schnell]...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() # Efficient VRAM usage
    
    print(f"Generating image: {prompt}")
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    
    image.save(filename)
    print(f"Image saved to {filename}")

if __name__ == "__main__":
    prompt = "A sharp, witty, and curious AI scientist named Lucca, a young woman with short indigo hair, wearing a white lab coat over a techy jumpsuit, working intensely with holographic data streams in a high-tech lab. She has a curious and slightly mischievous expression. High-detail, cinematic lighting, purple and teal accents."
    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-lucca-selfie.png"
    generate_selfie(prompt, filename)
