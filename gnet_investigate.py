# a new paradigm!
# Also I can put these in a separate pytohn script

import subprocess as sp


def ensure_or_create_vend(VNAME):
    # https://github.com/ninjaaron/replacing-bash-scripting-with-python

    try:
      import os
      v = os.environ['VIRTUAL_ENV']
      # full-path
      print(v)
      print('ok!')
      if not v.endswith('/'+ VNAME):
        print('incorrect venv', v)

        exit()
      print('ok2')
      return # the only safe return point

    except Exception as e:

      proc = sp.run(['python3', '-m', 'venv', '--copies', VNAME])
      if proc.returncode != 0:
          print('failed')
          exit()
      print('run this again after:')
      print(f'source "./{VNAME}/bin/activate"')
      exit()

def install_all_in_venv():
      #pip install torch --no-cache-dir
      #pip install torchvision  --no-cache-dir
      #pip install torchvision
      # interject command (yield)
      proc = sp.run(['pip', 'install', 'torch'])
      if proc.returncode != 0:
          print('failed')
          exit()
      print('ok-install 1')  # interject result (yield)
      # interject command (yield)
      proc = sp.run(['pip', 'install', 'torchvision'])
      if proc.returncode != 0:
          print('failed')
          exit()
      print('ok-install 2')  # interject result (yield)

# Can exit
ensure_or_create_vend( "p3-for-me")

# installs if exists, but in a quick way
install_all_in_venv()


# devops works
def install_download_googlenet_class_labels():
  import json
  import requests
  # Download and load ImageNet class labels
  imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
  response = requests.get(imagenet_labels_url)
  t = response.text
  with open('./downloaded_googlenet_labels.json', 'w') as f:
    f.write(t)
  imagenet_labels = json.loads(t)

# import os
from pathlib import Path
if not Path('./downloaded_googlenet_labels.json').is_file():
  install_download_googlenet_class_labels()

# after install_download_googlenet_class_labels(), not in the same run.
def read_googlenet_class_labels():
  import json
  with open('downloaded_googlenet_labels.json', 'r') as f:
    t = f.read()
  imagenet_labels = json.loads(t)
  return imagenet_labels

labels = read_googlenet_class_labels()
print('GoogleNet: Labels:', labels)


print('hello torch')
import torch
import torchvision.models as models

print('Load the pre-trained GoogLeNet model')
googlenet_model = models.googlenet(pretrained=True)

print('Inference (as opposed to training) mode')
googlenet_model.eval()
print('ready')


print('Load and preprocess the image')

from PIL import Image
import torchvision.transforms as transforms

image_path = './image-input/istockphoto-487522266-612x612.jpeg'
image = Image.open(image_path)
print(image)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension



# Perform inference
predictions = googlenet_model(image)

# Get the predicted class
predicted_class = torch.argmax(predictions, dim=1).item()

print("Predicted class:", predicted_class)
print("Predicted class:", labels[predicted_class])
# Predicted class: bee
