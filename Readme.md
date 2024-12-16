We implement our method in agents/ppo.py
## Setup
### Requirements
- Ubuntu 20.04
- Python 3.7
- CUDA >=11.0


### Procgen
```bash
pip install procgen
```
### Python module requirements
```bash
conda env create -f environment.yaml
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensofrlow-gpu==2.5.1 
pip install gym==0.15.3
pip install higher==0.2 kornia==0.3.0
pip install tensorboard termcolor matplotlib imageio imageio-ffmpeg 
pip install scikit-image pandas pyyaml
```

## Training
```bash

env_name1="bigfish"


seed1=1


alpha=1e-2

norm1=1


adversarial_repeat=2

#FGA3
python train.py --algo ppo --env_name $env_name1 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat


#For SAR
python train.py --algo sar --env_name $env_name1 --seed $seed1
```
