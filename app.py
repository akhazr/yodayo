from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    
    # this will substitute the default PNDM scheduler for K-LMS  
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    model = StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='hakurei/waifu-diffusion', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "muffins"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
