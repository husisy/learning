import os
import dotenv

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

dotenv.load_dotenv()

project_id = os.environ['google_project_id']
location = os.environ['google_location']
vertexai.init(project=project_id, location=location)

multimodal_model = GenerativeModel("gemini-pro-vision")

response = multimodal_model.generate_content(
    [
        "what is shown in this image?",
        Part.from_uri(
            "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
        ),
    ]
)
print(response)
multimodal_model = GenerativeModel("gemini-pro-vision")
