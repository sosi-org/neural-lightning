# a new paradigm!

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
    except Exception as e:

      import subprocess as sp
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
      proc = sp.run(['pip', 'install', 'torchvision'])
      if proc.returncode != 0:
          print('failed')
          exit()
      proc = sp.run(['pip', 'install', 'torchvision'])
      if proc.returncode != 0:
          print('failed')
          exit()
      print('ok')

      exit()
# Can exit
ensure_or_create_vend( "p3-for-me")
exit()
install_all_in_venv()

exit()
import torch
import torchvision.models as models

# Load the pre-trained GoogLeNet model
googlenet_model = models.googlenet(pretrained=True)

googlenet_model.eval()

