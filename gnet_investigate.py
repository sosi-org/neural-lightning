# neural-lightning

# devops: self-installer:
# a new paradigm! Use the same language
# Also I can put these in a separate pytohn script
# standalone: just run it anywhere

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

      # todo: input from another one
      # interject command (yield)
      proc = sp.run(['pip', 'install', 'matplotlib'])
      if proc.returncode != 0:
          print('failed')
          exit()
      print('ok-install 3')  # interject result (yield)

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

googlenet_labels = read_googlenet_class_labels()
# print('GoogleNet: All Labels:', googlenet_labels)


print('hello torch')
import torch
import torchvision.models as models

print('Load the pre-trained GoogLeNet model')
googlenet_model = models.googlenet(pretrained=True)

print('Inference (as opposed to training) mode')
googlenet_model.eval()
print('ready')


print('Load and preprocess the image')


def process_image(image_path, intercept=None):

  from PIL import Image
  import torchvision.transforms as transforms

  image = Image.open(image_path)
  # print(image)

  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  image = transform(image)
  image = image.unsqueeze(0)  # Add batch dimension

  if intercept is not None:
     intercept(googlenet_model, image)

  # Perform inference
  predictions = googlenet_model(image)

  # Get the predicted class
  predicted_class = torch.argmax(predictions, dim=1).item()

  # print("Predicted class:", predicted_class)
  # print("Predicted class:", googlenet_labels[predicted_class])

  # Predicted class: bee
  return predicted_class, googlenet_labels[predicted_class]


def visualise_me(nparray_3d):
    """
    nparray_3d: [W,H,3]  , where 3 is for R,G,B
    """
    # torch.Size([1, 1024, 7, 7])
    # from utils.pcolor import PColor
    from pcolor import PColor

    # PColor.plot_show_image(G_paintings2d, file_id, sleep_sec, more_info)
    PColor.plot_show_image(nparray_3d, 'file_id', 1, ('more_info', 'info2'))

def intercept(googlenet_model, image):
    # Define a hook to store the intermediate layer outputs
    intermediate_outputs = []

    def hook(module, input, output):
        intermediate_outputs.append(output)

    # Register the hook for a specific layer (example: mixed_5b)
    layer_to_trace = googlenet_model.inception5b
    handle = layer_to_trace.register_forward_hook(hook)

    # See note [1]
    # Perform inference
    _ = googlenet_model(image)

    # Remove the hook
    handle.remove()

    # Print the intermediate outputs
    for idx, output in enumerate(intermediate_outputs):
        print(f"Output {idx}: {output.shape}")
        print(output)
        # torch.Size([1, 1024, 7, 7])

        # output.cpu().numpy()  --> Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        npa = output.cpu().detach().numpy()
        print('npa.shape', npa.shape) # (1, 1024, 7, 7)
        npa2 = npa[0,:,:,:3]
        npa3 = npa2.copy()
        visualise_me(npa3)

import glob
for image_path in glob.glob('./image-input/**/*'):
    class_id, class_name = process_image(image_path, intercept)
    print(class_id, class_name, ' <-- ', image_path)
    # exit()
    break

def show_module_info():
    """
    In the PyTorch implementation of GoogLeNet, you can examine the details of the inception5b module by simply printing it:
    """

    import torchvision.models as models
    googlenet_model = models.googlenet(pretrained=True)
    print(googlenet_model.inception5b)

show_module_info()

"""
    Note [1]
    In GoogLeNet, the inception5b module is an Inception module, which is a substructure designed to capture local features at different scales by applying multiple parallel convolutional filters with different kernel sizes. Inception modules help increase the network's capability to capture both high-level and low-level features, while keeping computational complexity manageable.

    The inception5b module is the last Inception module in the GoogLeNet architecture. Its structure consists of the following branches:

    Branch 1: 1x1 convolution
    Branch 2: 1x1 convolution followed by a 3x3 convolution
    Branch 3: 1x1 convolution followed by a 5x5 convolution
    Branch 4: 3x3 max-pooling followed by a 1x1 convolution


    Each branch operates in parallel, and their outputs are concatenated along the channel axis to form the final output of the Inception module.

    Here's an outline of the inception5b module structure:

    inception5b
    ├── 1x1 Convolution
    ├── 3x3 Branch
    │   ├── 1x1 Convolution
    │   └── 3x3 Convolution
    ├── 5x5 Branch
    │   ├── 1x1 Convolution
    │   └── 5x5 Convolution
    └── Max-Pooling Branch
        ├── 3x3 Max-Pooling
        └── 1x1 Convolution

"""
