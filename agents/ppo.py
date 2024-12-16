import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import numpy as np
import sys 
from collections import deque

class Ppo():
    """
    Ppo
    """
    def __init__(self,
                 args,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 eps,
                 max_grad_norm):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.step = 0
        self.args=args

    def make_adversarial_samples(self,obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ):

        obs_batch_aug=obs_batch.clone()

        spectrum = torch.fft.fftn(obs_batch_aug, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

        # Get amplitude and Phase
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        a=[]
        from tqdm import tqdm
        for i in tqdm(range(10000)):

            obs_amplitude=amplitude.clone().detach().requires_grad_(True)
            obs_complex = obs_amplitude * torch.exp(1j * torch.angle(spectrum))
            obs_complex = torch.fft.ifftshift(obs_complex, dim=(-2, -1))
            obs_complex = torch.fft.ifftn(obs_complex, dim=(-2, -1)).float()

            values, action_log_probs, dist_entropy, logits, _ = self.actor_critic.evaluate_actions(
            obs_complex, recurrent_hidden_states_batch, masks_batch,
            actions_batch)


            value_losses = (values - return_batch).pow(2)


            adversarial_loss = 0.5 * value_losses.mean()


            x= self.actor_critic.get_features(
            obs_batch, recurrent_hidden_states_batch, masks_batch,
            actions_batch).detach()
            y= self.actor_critic.get_features(
            obs_complex, recurrent_hidden_states_batch, masks_batch,
            actions_batch)
            #print(x.shape,y.shape)
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            semantic_distance = 2 - 2 * (x * y).sum(dim=-1).mean()
            gradients = torch.autograd.grad(adversarial_loss - self.args.adversarial_alpha * semantic_distance, obs_amplitude)[
                0]
            norm = torch.norm(gradients, p=2)
            gradients = 1e10 * gradients / (norm + 1e-24)  # Add a small constant to avoid division by zero
            a.append(adversarial_loss.item())
            amplitude = amplitude + gradients
            amplitude = torch.clamp(amplitude, min=0.0)
            obs_aug = amplitude * torch.exp(1j * phase)
            obs_aug = torch.fft.ifftshift(obs_aug, dim=(-2, -1))
            obs_aug = torch.fft.ifftn(obs_aug, dim=(-2, -1)).float()
            obs_aug = torch.clamp(obs_aug, min=0.0)

        import matplotlib.pyplot as plt
        plt.plot(a)
        plt.show()
        exit()

        return obs_aug,recurrent_hidden_states_batch,actions_batch,value_preds_batch,return_batch,masks_batch,old_action_log_probs_batch,adv_targ
    def update(self, rollouts):
        self.step += 1

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Make Adversarial Sample
                obs_aug, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ=self.make_adversarial_samples(obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ)

                # Calculate Value and Action Log Probability of Original Data
                values, action_log_probs, dist_entropy, logits, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                      actions_batch)

                # Calculate Value and Action Log Probability of Adversarial Data
                adv_values, adv_action_log_probs, adv_dist_entropy, logits, _ = self.actor_critic.evaluate_actions(
                    obs_aug, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)


                # Value Loss of Original
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()

                # Value Loss of Adversarial
                adv_value_losses = (adv_values - return_batch).pow(2)
                adv_value_loss = 0.5 * torch.max(adv_value_losses,
                                                value_losses_clipped).mean()


                # Actor Loss of Original
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                # Update actor-critic using both PPO Loss

                # Actor Loss of Adversarial
                adv_ratio = torch.exp(adv_action_log_probs -
                                  old_action_log_probs_batch)
                adv_targ = return_batch- adv_values
                adv_targ = (adv_targ - adv_targ.mean()) / (
                        adv_targ.std() + 1e-5)
                surr1 = adv_ratio * adv_targ
                surr2 = torch.clamp(adv_ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                adv_action_loss = -torch.min(surr1, surr2).mean()



                self.optimizer.zero_grad()
                loss = action_loss+adv_action_loss + self.value_loss_coef *(value_loss+adv_value_loss) - self.entropy_coef * (dist_entropy+adv_dist_entropy)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                    
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
