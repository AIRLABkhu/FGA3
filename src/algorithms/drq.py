import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC
from augmentations import *

class DrQ(SAC): # [K=1, M=1]
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.args=args
	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()


		# If want to run original DrQ eliminate this line.
		obs, action, reward, next_obs, not_done=self.make_adversarial_samples(obs, action, reward,
																		 next_obs, not_done, L=None, step=None)

		obs=random_shift(obs,None,4)
		next_obs=random_shift(next_obs,None,4)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
