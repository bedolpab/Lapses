import requests
import shutil
import numpy as np
import 

url = "https://thispersondoesnotexist.com/image"
response  = requests.get(url, stream=True)

print(response)
