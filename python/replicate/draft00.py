import dotenv
import replicate
import requests


dotenv.load_dotenv()

## stable-diffusion
z0 = replicate.run("stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        input={"prompt": "a 19th century portrait of a wombat gentleman"})
z1 = requests.get(z0[0])
# https://replicate.delivery/pbxt/emPecibeFaH7PJBxv7y0zf6M0uWGp1dTiYjASuP5lGbiEGJEB/out-0.png


## resnet


## text2image
z0 = replicate.run(
  "pixray/text2image:5c347a4bfa1d4523a58ae614c2194e15f2ae682b57e3797a5bb468920aa70ebf",
  input={"prompts": "robots talking to robots"},
)
# z1 = list(z0) #longer than 5 minute
