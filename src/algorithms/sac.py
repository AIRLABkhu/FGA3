import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m


class SAC(object):
	def __init__(self, obs_shape, action_shape, args):
		self.device=torch.device("cuda:{}".format(args.gpu))
		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq
		self.args=args
		shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).to(self.device)
		head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).to(self.device)
		actor_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		critic_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)

		self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).to(self.device)
		self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).to(self.device)
		self.critic_target = deepcopy(self.critic)

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)

		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)

		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def eval(self):
		self.train(False)

	@property
	def alpha(self):
		return self.log_alpha.exp()
		
	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).to(self.device)
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
		return mu.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
		return pi.cpu().data.numpy().flatten()

	def sample_heatmap(self,obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			output = self.actor.show_headmap(_obs)
		return output.detach().cpu().numpy()

	def sample_latent(self,obs):
		output = self.actor.encoder(obs)

		return output

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)



		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1,
								 target_Q) + F.mse_loss(current_Q2, target_Q)

		#semantic_loss=nn.MSELoss(reduction='sum')
		#semantic_distance=0.5*semantic_loss(self.critic.encoder(obs[self.args.batch_size:,:,:,:]).detach(),self.critic.encoder(obs[:self.args.batch_size,:,:,:]))
		#critic_loss-=self.args.adversarial_alpha*semantic_distance

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		_, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

	def soft_update_critic_target(self):
		utils.soft_update_params(
			self.critic.Q1, self.critic_target.Q1, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.Q2, self.critic_target.Q2, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.encoder, self.critic_target.encoder,
			self.encoder_tau
		)


	def get_adversarial_loss(self, obs, action, reward, next_obs, not_done, L=None, step=None):

		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1,
								 target_Q) + F.mse_loss(current_Q2, target_Q)

		return critic_loss
	def make_adversarial_samples(self, obs, action, reward, next_obs, not_done, L=None, step=None):

		# copy
		obs_aug=obs.clone()

		# Fourier Transform
		spectrum=torch.fft.fftn(obs_aug,dim=(-2,-1))
		spectrum=torch.fft.fftshift(spectrum,dim=(-2,-1))

		# Get amplitude and Phase
		amplitude=torch.abs(spectrum)
		phase=torch.angle(spectrum)
		semantic_loss=nn.MSELoss(reduction='sum')
		for i in range(self.args.adversarial_repeat):

			# Compute adversarial loss with respect to amplitude
			obs_amplitude = amplitude.clone().detach().requires_grad_(True)  # Clone amplitude and enable gradient computation
			obs_complex = obs_amplitude * torch.exp(1j * torch.angle(spectrum))
			obs_complex=torch.fft.ifftshift(obs_complex,dim=(-2,-1))
			obs_complex=torch.fft.ifftn(obs_complex,dim=(-2,-1)).float()

			adversarial_loss = self.get_adversarial_loss(obs_complex, action, reward, next_obs, not_done, L=None,
														 step=None)
			x=self.critic.encoder(obs).detach()
			y=self.critic.encoder(obs_complex)
			
			x=F.normalize(x,dim=-1,p=2)
			y=F.normalize(y,dim=-1,p=2)
			
			semantic_distance=2-2*(x*y).sum(dim=-1).mean()
			#semantic_distance=0.5*semantic_loss(self.critic.encoder(obs).detach(),self.critic.encoder(obs_complex))

			
			# Compute gradients with respect to amplitude only
			gradients = torch.autograd.grad(adversarial_loss-self.args.adversarial_alpha*semantic_distance, obs_amplitude)[0]
			norm = torch.norm(gradients, p=2)
			gradients = self.args.norm*gradients / (norm + 1e-24)  # Add a small constant to avoid division by zero

			amplitude=amplitude+gradients
			amplitude = torch.clamp(amplitude,min=0.0)
			obs_aug = amplitude * torch.exp(1j*phase)
			obs_aug=torch.fft.ifftshift(obs_aug,dim=(-2,-1))
			obs_aug=torch.fft.ifftn(obs_aug,dim=(-2,-1)).float()
			obs_aug=torch.clamp(obs_aug,min=0.0)



		# Make Adversarial Batch : {B_ori,B_adcv}
		obs=torch.cat([obs,obs_aug],dim=0)
		action=torch.cat([action,action],dim=0)
		reward=torch.cat([reward,reward],dim=0)
		next_obs=torch.cat([next_obs,next_obs],dim=0)
		not_done=torch.cat([not_done,not_done],dim=0)

		return obs,action,reward,next_obs,not_done



	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()


		#if self.args.adversarial==True:
		obs, action, reward, next_obs, not_done=self.make_adversarial_samples(obs, action, reward,
																			 next_obs, not_done, L=None, step=None)



		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()


