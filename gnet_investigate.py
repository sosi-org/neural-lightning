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



def visualise_me(nparray_3d_arr):
    """
    nparray_3d: [W,H,3]  , where 3 is for R,G,B
    """
    # torch.Size([1, 1024, 7, 7])
    # from utils.pcolor import PColor
    from pcolor import PColor

    # PColor.plot_show_image(G_paintings2d, file_id, sleep_sec, more_info)
    # PColor.plot_show_image(nparray_3d, 'file_id', 1, ('more_info', 'info2'))
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    plt.clf()
    fig, axes = plt.subplots(7,7)
    print(axes, '00000')
    for i in range(len(nparray_3d_arr)):
        print('i',i)
        # plt.subplot(7,7, i+1)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        import matplotlib
        matplotlib.rc('axes', edgecolor='white')
        matplotlib.rc('axes', facecolor='black')

        nparray_3d = nparray_3d_arr[i]
        scaled_back_to_255 = nparray_3d * 127.0 + 128
        #scaled_back_to_255 = nparray_3d * 127.0 + 128
        #scaled_back_to_255[scaled_back_to_255 > 255] = 255
        #scaled_back_to_255[scaled_back_to_255 <0 ] = 0
        #plt.imshow(scaled_back_to_255.astype(np.uint8))

        # unscaled: in range: (0,5)
        ic = i % 7
        jc = int(i / 7)
        axes[ic][jc].imshow(nparray_3d)
        #  axes[i]
        # plt.colorbar()
    # plt.draw()
    plt.show()

def visu_all_4d_tensor(output):
    print(output)
    # torch.Size([1, 1024, 7, 7])

    # output.cpu().numpy()  --> Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    npa = output.cpu().detach().numpy()
    print('npa.shape', npa.shape) # (1, 1024, 7, 7)

    #npa2 = npa[0,:,:,:3]
    #npa3 = npa2.copy()
    ## visualise_me(npa3)

    img_array = []
    for i in range(npa.shape[2]):
      for j in range(npa.shape[3]):
        n1000 = npa[0,:,i,j]
        n1000 = n1000.reshape((32,32))[:,:,None]
        img_array.append(n1000)
    visualise_me(img_array)

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
        visu_all_4d_tensor(output)


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

torch.Size([1, 1024, 7, 7]):
1 is batch size

The second dimension (1024) represents the number of feature maps (channels) produced by the inception5b layer. Each feature map captures specific features or patterns in the input image. In GoogLeNet's Inception modules, the number of feature maps is determined by the concatenation of the outputs from different branches (1x1, 3x3, 5x5, and max-pooling branches).

The third and fourth dimensions (7, 7) represent the spatial dimensions (height and width) of the feature maps. At this point in the GoogLeNet architecture, the spatial dimensions of the input image have been reduced significantly due to successive convolutional and pooling layers. The 7x7 feature maps contain spatial information about the detected features or patterns in the input image.

In summary, the output tensor contains 1024 feature maps, each of size 7x7, for a single input image. Each feature map encodes specific information about the input image, which is then used by subsequent layers in the network for higher-level reasoning and classification.


1024:
The 1024 feature maps in the inception5b module's output are a result of concatenating the outputs of each of its branches. Here's the breakdown of the 1024 feature maps for each branch in the inception5b module:

1x1 Convolution: 384 feature maps

3x3 Branch:
1x1 Convolution (input): 192 feature maps
3x3 Convolution (output): 384 feature maps

5x5 Branch:
1x1 Convolution (input): 48 feature maps
5x5 Convolution (output): 128 feature maps

Max-Pooling Branch:
3x3 Max-Pooling (input)
1x1 Convolution (output): 128 feature maps
When you concatenate the outputs of each branch, you get a total of 384 + 384 + 128 + 128 = 1024 feature maps.

Note that the number of feature maps produced by each branch is a design choice, and these values were determined through experimentation by the authors of the GoogLeNet architecture. The goal is to balance the computational complexity and the model's capability to capture various features at different scales.


1x1 Convolution: 384 feature maps
3x3 Branch: 384 feature maps
5x5 Branch: 128 feature maps
Max-Pooling Branch: 128 feature maps

[] See: Going deeper with convolutions: https://arxiv.org/pdf/1409.4842v1.pdf

"""
