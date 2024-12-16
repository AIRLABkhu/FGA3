# FGA3
 Official Pytorch Implementation ``Fourier Guided Adaptive Adversarial Augmentation for Generalization in Visual Reinforcement Learning'' 

## Abstract
> Visual Reinforcement Learning (RL) facilitates learning directly from raw images; however, the domain gap between training and testing environments frequently leads to a decline in performance within unseen environments.
In this paper, we propose Fourier Guided Adaptive Adversarial Augmentation (FGA3), a novel augmentation method that maintains semantic consistency.
We focus on style augmentation in the frequency domain by keeping the phase and altering the amplitude to preserve the state of the original data.
For adaptive adversarial perturbation, we reformulate the worst-case problem to RL by employing adversarial example training, which leverages value loss and cosine similarity within a semantic space.
Moreover, our findings illustrate that cosine similarity is effective in quantifying feature distances within a semantic space.
Extensive experiments on DMControl-GB and Procgen have shown that FGA3 is compatible with a wide range of visual RL algorithms, both off-policy and on-policy, and significantly improves the robustness of the agent in unseen environments.

## Framework
![Screenshot from 2024-12-16 10-48-50](https://github.com/user-attachments/assets/f2d81edb-9dbd-4d26-b3a0-3e460b52b3eb)



## Experimental Results
### DMControl-GB
![Screenshot from 2024-12-16 10-48-57](https://github.com/user-attachments/assets/ff7a0bb4-7971-4f8d-b84f-dc976c20fa73)

### Procgen
![Screenshot from 2024-12-16 10-49-02](https://github.com/user-attachments/assets/f564fb47-c1bb-4083-8587-78a16f6c6c93)

## Setup
### Install MuJoCo
Download the MuJoCo version 2.0 binaries for Linux or OSX. 


### Install DMControl

``` bash
conda env create -f setup/conda.yaml
conda activate fga3
sh setup/install_envs.sh
```

### Usage
``` bash
from env.wrappers import make_env  
env = make_env(  
        domain_name=args.domain_name,  
        task_name=args.task_name,  
        seed=args.seed,  
        episode_length=args.episode_length,  
        action_repeat=args.action_repeat,  
        image_size=args.image_size,  
        mode='train'  
)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  
```

You can try other environments easily.



### Training
``` bash
python src/train.py --domain_name walker --task_name walk --algorithm sac --seed 2087 --eval_episode 30 --action_repeat 4 --adversarial_repeat 8 --adversarial_alpha 1e-2 --norm 1e5 --gpu 0
```

### Contact
For any questions, discussions, and proposals, please contact us at everyman123@khu.ac.kr

### Code Reference
- https://github.com/Yara-HYR/SRM

 
