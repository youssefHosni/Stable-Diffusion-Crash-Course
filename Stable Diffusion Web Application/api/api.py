from auth_token import auth_token
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
print(torch.cuda.is_available())
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from io import BytesIO
import base64 


# create the Fast API instance and set the configurations 
app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)


device = torch.device('cuda')
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) #load the stable diffusion pipeline 
pipe.safety_checker = lambda images, clip_input: (images, False) # to stop the deteecting NSFW
pipe.enable_attention_slicing()
pipe.to(device)

# define Fast API endpoint that generate an image based on a given prompt string. 

@app.get("/")
def generate(prompt: str): 
 
    image = pipe(prompt, 
                 guidance_scale=7,  # Forces generation to better match the prompt, 7 or 8.5 give good results, results are better the larger the number is, but will be less diverse
                 height=64, 
                 width=64).images[0]

    image.save("testimage.png") # save the image

    # return the image
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())
    return Response(content=imgstr, media_type="image/png")