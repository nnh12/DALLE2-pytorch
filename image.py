from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Ensure you have a GPU to run this

def generate_image_from_text(prompt):
    image = pipe(prompt).images[0]
    image.show()  # Display the image
    image.save("output_image.png")  # Save the image locally

# Example usage
prompt = "A serene forest with a flowing river during autumn"
generate_image_from_text(prompt)

