from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipe.enable_attention_slicing()

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars ona purple sky background"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse2.png")