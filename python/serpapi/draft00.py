import os
import dotenv
from serpapi import GoogleSearch

dotenv.load_dotenv()
APIKEY = os.environ['SERPAPI']

search = GoogleSearch({
    "q": "coffee",
    "location": "Austin,Texas",
    "api_key": APIKEY,
  })
result = search.get_dict()
search.get_account()
