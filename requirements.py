import subprocess
import sys

subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

subprocess.run(
  ['pip', 'install', 'git+https://github.com/facebookresearch/detectron2.git'])
