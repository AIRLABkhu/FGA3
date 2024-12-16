
env_name1="bigfish"
env_name2="starpilot"
env_name3="fruitbot"
env_name4="bossfight"
env_name5="ninja"
env_name6="plunder"
env_name7="caveflyer"
env_name8="coinrun"
env_name9="jumper"
env_name10="chaser"
env_name11="climber"
env_name12="dodgeball"
env_name13="heist"
env_name14="leaper"
env_name15="maze"
env_name16="miner"

seed1=1


alpha=1e-2

norm1=10


adversarial_repeat=2

#FGA3

python train.py --algo ppo --env_name $env_name7 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name8 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name9 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name10 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name11 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name12 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name13 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name14 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name15 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat
python train.py --algo ppo --env_name $env_name16 --seed $seed1 --norm $norm1 --tag $norm1 --adversarial_repeat $adversarial_repeat

