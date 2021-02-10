import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
import math

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES


class SimilarityRoleLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.inc_mask = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, self.n_agents).to(self.device)
        self.inc_mask_actions = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, self.n_agents, 1).to(self.device)

        self.env_sim_mask = (1 - th.eye(self.n_agents)).reshape(1,1,self.n_agents,self.n_agents,1).to(self.device)
        self.inc_sim_mask = (1 - th.eye(self.n_agents)).reshape(1,1,self.n_agents,1,self.n_agents).to(self.device)
        # [bs,t-1,n,n,n] # of i, k choose, to j, Q


        self.inc_scale = 1  # self.n_agents
        self.inc_cost_scale = 1  # /(self.n_agents - 1)

        self.sim_horizon = args.sim_horizon

        self.env = args.env

        self.params = list(mac.parameters())
        self.params_env = mac.parameters_env()
        self.params_inc = mac.parameters_inc()
        self.params_role = mac.parameters_role()

        # assert len(self.params_env) + len(self.params_inc) + len(self.params_role)- 4 == len(self.params), "parameters missed"

        self.last_target_update_episode = 0

        # self.optimiser = Adam(params=self.params, lr=args.lr)
        self.optimiser_env = Adam(params=self.params_env, lr=args.lr_env)
        self.optimiser_inc = Adam(params=self.params_inc, lr=args.lr_inc)
        self.optimiser_role = Adam(params=self.params_role, lr=args.lr_role)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def cal_loss_and_step(self, batch: EpisodeBatch):
        learning_logs = {}
        # ********************************************* data from buffer ***********************************************
        rewards = batch["reward"][:, :-1] / self.args.reward_scale  # [bs,t-1,n]
        # rewards_collective = rewards.sum(dim=-1, keepdim=True)  # [bs,t-1,n]
        actions = batch["actions"][:, :-1]  # [bs,t-1,n,1]
        actions_inc = batch["actions_inc"][:, :-1]  # [bs,t-1,n,n,1]
        actions_inc_all = batch["actions_inc"]  # [bs,t,n,n,1]
        # actions_inc_last_step = th.cat([actions_inc[:, 0].unsqueeze(1), actions_inc[:, :-1]], dim=1)

        clean_num = batch["clean_num"][:, :-1].clone()  # [bs,t-1,n]
        # clean_num /= (clean_num.max() + 1e-6)
        clean_num = (clean_num > 0).float()
        # apple_den = batch["apple_den"][:, :-1]  # [bs,t-1,n]

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # [bs,t-1,1]
        avail_actions = batch["avail_actions"]

        # ************************************************ mac out *****************************************************
        q_env = []
        q_inc = []
        # loss_identifiable = 0
        # role = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q_env_t, q_inc_t, extra_return = self.mac.forward(batch, t=t)
            q_env.append(q_env_t)  # [bs,n,a_env]
            q_inc.append(q_inc_t)  # [bs,n,n,a_inc]
            # loss_identifiable += extra_return['loss_identifiable']
            # role.append(extra_return['role']) # [bs,n,role]
        q_env = th.stack(q_env, dim=1)  # [bs,t,n,a_env]
        q_inc = th.stack(q_inc, dim=1)  # [bs,t,n,n,a_inc]
        # role = th.stack(role, dim=1)  # [bs,t,n,role]
        # loss_identifiable /= batch.max_seq_length

        avail_inc_actions = th.ones_like(q_inc) # [bs,t,n,n,a_inc]
        if self.args.disable_positive_inc_action:
            avail_inc_actions[:,:,:,:,1] = 0
        elif self.args.disable_negative_inc_action:
            avail_inc_actions[:,:,:,:,2] = 0

        # ********************************************** target mac out ************************************************
        target_q_env = []
        target_q_inc = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q_env_t, q_inc_t, extra_return = self.target_mac.forward(batch, t=t)  # [bs,n,-1]
            target_q_env.append(q_env_t.detach())
            target_q_inc.append(q_inc_t.detach())

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_q_env = th.stack(target_q_env[1:], dim=1)  # [bs,t,n,a_env]
        target_q_inc = th.stack(target_q_inc[1:], dim=1)  # [bs,t,n,n,a_inc]

        # ******************************************** inc rewards *****************************************************
        effect_ratio = self.args.incentive_ratio
        cost_ratio = self.args.incentive_cost

        # 0,1,2 = NO, +, -
        actions_inc_masked = actions_inc * self.inc_mask_actions  # * rewards.unsqueeze(-1).unsqueeze(-1) # [bs,t-1,n,n,1]
        actions_inc_all_masked = actions_inc_all * self.inc_mask_actions  # [bs,t,n,n,1]
        # give_value = []
        # receive_value = []
        #
        # for i in range(self.n_agents):  # [bs,t-1,n,n,1]
        #     i_give = th.sum(actions_inc_masked[:, :, i] != 0, dim=(2, 3))  # [bs,t-1]
        #
        #     i_receive = th.sum(actions_inc_masked[:, :, :, i] == 1, dim=(2, 3)) \
        #                 - th.sum(actions_inc_masked[:, :, :, i] == 2, dim=(2, 3))  # [bs, t-1]
        #     give_value.append(i_give)
        #     receive_value.append(i_receive)
        # give_value = th.stack(give_value, dim=-1)  # [bs,t-1,n]
        # receive_value = th.stack(receive_value, dim=-1)  # [bs,t-1,n]

        give_value = (actions_inc_masked != 0).sum(dim=(3,4)) # [bs,t-1,n]
        receive_positive_all = (actions_inc_all_masked == 1).sum(dim=(2, 4))  # [bs,t,n]
        receive_negative_all = (actions_inc_all_masked == 2).sum(dim=(2, 4))  # [bs,t,n]
        receive_zero_all = self.n_agents - 1 - receive_positive_all - receive_negative_all
        receive_positive = receive_positive_all[:, :-1]  # [bs,t-1,n]
        receive_negative = receive_negative_all[:, :-1]  # [bs,t-1,n]
        receive_zero = receive_zero_all[:, :-1]  # [bs,t-1,n]
        receive_value = receive_positive - receive_negative

        rewards_inc = receive_value * effect_ratio - give_value * cost_ratio  # [bs,t-1,n]

        rewards_reassigned = (rewards + rewards_inc * self.args.incentive) / (batch.max_seq_length)  # [bs,t-1,n]

        rewards_for_env = (rewards + receive_value * effect_ratio * self.args.incentive) / (batch.max_seq_length)  # [bs,t-1,n]
        rewards_for_inc = (rewards - give_value * cost_ratio * self.args.incentive) / (batch.max_seq_length)  # [bs,t-1,n]

        # **************************************** value loss **********************************************************
        chosen_action_env_qvals = th.gather(q_env[:, :-1], dim=-1, index=actions)  # [bs,t,n,a_env] ==> [bs,t-1,n,1]
        if self.args.consider_others_inc:
            chosen_action_inc_qvals_other = (q_inc[:, :-1, :, :, 0] * receive_zero.unsqueeze(2) +
                                             q_inc[:, :-1, :, :, 1] * receive_positive.unsqueeze(2) +
                                             q_inc[:, :-1, :, :, 2] * receive_negative.unsqueeze(2))  # /(self.n_agents - 1)
            chosen_action_inc_qvals = chosen_action_inc_qvals_other/(self.n_agents - 1)
        else:
            chosen_action_inc_qvals_self = th.gather(q_inc[:, :-1], dim=-1, index=actions_inc).squeeze(-1)  # [bs,t,n,n,a_inc] ==> [bs,t-1,n,n]
            chosen_action_inc_qvals = chosen_action_inc_qvals_self

        # chosen_action_qvals = th.cat(
        #     [chosen_action_env_qvals, chosen_action_inc_qvals * self.inc_mask / (self.inc_scale)],
        #     # np.sqrt(self.n_agents)
        #     dim=-1)  # [bs,t-1,n,1+n]

        # Mask out unavailable actions
        target_q_env[avail_actions[:, 1:] == 0] = - 9999999
        target_q_inc[avail_inc_actions[:, 1:] ==0] = - 9999999

        target_max_qvals_inc_other = (target_q_inc[:, :, :, :, 0] * receive_zero_all[:, 1:].unsqueeze(2) +
                                      target_q_inc[:, :, :, :, 1] * receive_positive_all[:, 1:].unsqueeze(2) +
                                      target_q_inc[:, :, :, :, 2] * receive_negative_all[:, 1:].unsqueeze(2))
        target_next_qvals_inc = th.gather(target_q_inc, dim=-1, index=actions_inc_all[:, 1:]).squeeze(-1)  # [bs,t-1,n,n]

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            q_env_detach = q_env.clone().detach()
            q_inc_detach = q_inc.clone().detach()

            q_env_detach[avail_actions == 0] = - 9999999
            q_inc_detach[avail_inc_actions == 0] = - 9999999

            cur_max_actions_env = q_env_detach[:, 1:].max(dim=-1, keepdim=True)[1]  # [bs,t-1,n,1]
            cur_max_actions_inc = q_inc_detach[:, 1:].max(dim=-1, keepdim=True)[1]  # [bs,t-1,n,n,1]

            target_max_qvals_env = th.gather(target_q_env, dim=-1, index=cur_max_actions_env)  # [bs,t-1,n,1]
            target_max_qvals_inc_self = th.gather(target_q_inc, dim=-1, index=cur_max_actions_inc).squeeze(-1)  # [bs,t-1,n,n]
            if self.args.consider_others_inc:
                target_max_qvals_inc = (target_max_qvals_inc_self + target_max_qvals_inc_other - target_next_qvals_inc) / (
                            self.n_agents - 1)
            else:
                target_max_qvals_inc = target_max_qvals_inc_self
        else:
            target_max_qvals_env = target_q_env.max(dim=-1)[0]  # [bs,t-1,n,1]
            target_max_qvals_inc_self = target_q_inc.max(dim=-1)[0].squeeze(-1)  # [bs,t-1,n,n]
            # target_max_qvals_inc = target_max_qvals_inc_self
            if self.args.consider_others_inc:
                target_max_qvals_inc = (target_max_qvals_inc_self + target_max_qvals_inc_other - target_next_qvals_inc) / (
                            self.n_agents - 1)
            else:
                target_max_qvals_inc = target_max_qvals_inc_self

            # target_max_qvals = th.cat(
            #     [target_max_qvals_env, target_max_qvals_inc * self.inc_mask / (self.inc_scale)],
            #     dim=-1)  # [bs,t-1,n,1+n]

        # TODOSSD: choose other mixers
        # chosen_action_qvals = chosen_action_qvals.sum(dim=-1)  # [bs,t-1,n,1+n] ==> [bs,t-1,n]
        # target_max_qvals = target_max_qvals.sum(dim=-1)  # [bs,t-1,n,1+n] ==>  [bs,t-1,n,1+n]

        # Calculate 1-step Q-Learning targets
        # targets = rewards_reassigned + self.args.gamma * (1 - terminated) * target_max_qvals
        # targets = rewards_reassigned + self.args.gamma * target_max_qvals # TODOSSD: test
        # # Td-error
        # td_error = (chosen_action_qvals - targets.detach())
        # mask = mask.expand_as(td_error)
        # # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask
        # # Normal L2 loss, take mean over actual data
        # value_loss = (masked_td_error ** 2).sum() / mask.sum()

        # # multi-step, 3 step # TODOSSD: test
        # targets = rewards_reassigned[:,0:-2] + self.args.gamma * rewards_reassigned[:,1:-1]  + \
        #                                        self.args.gamma**2 * rewards_reassigned[:,2:]  + \
        #                                        self.args.gamma**3 * target_max_qvals[:,2:]
        # td_error = (chosen_action_qvals[:,:-2] - targets.detach())
        # value_loss = (td_error**2).mean()

        # env & inc
        targets_env = rewards_for_env + self.args.gamma_env * (1 - terminated) * target_max_qvals_env.sum(dim=-1)
        targets_inc = rewards_for_inc + self.args.gamma_inc * (1 - terminated) * (
                    target_max_qvals_inc * self.inc_mask).sum(dim=-1)

        td_error_env = (chosen_action_env_qvals.sum(dim=-1) - targets_env.detach())
        td_error_inc = ((chosen_action_inc_qvals * self.inc_mask).sum(dim=-1) - targets_inc.detach())
        mask = mask.expand_as(td_error_env)
        masked_td_error_env = td_error_env * mask
        masked_td_error_inc = td_error_inc * mask
        value_loss_env = (masked_td_error_env ** 2).sum() / mask.sum()
        value_loss_inc = (masked_td_error_inc ** 2).sum() / mask.sum()

        # *********************************************** sim loss *****************************************************
        # h_sim [bs,t,n,n,dim] in i, j
        # actions_inc [bs,t-1,n,n,1], i->j
        # q_inc, [bs,t,n,n,a_inc] i->j
        # role [bs,t,n,role]

        # h_sim_ii = th.stack([h_sim[:,:,i,i,:] for i in range(self.n_agents)],dim=2).unsqueeze(3) # [bs,t,n,1,dim]
        # similarity = th.cosine_similarity(h_sim,h_sim_ii,dim=-1)[:,:-1].unsqueeze(-1) # [bs,t-1,n,n,1] # i,k

        # similarity = th.cosine_similarity(role.unsqueeze(dim=2),role.unsqueeze(dim=3),dim=-1)[:,:-1].unsqueeze(-1) # [bs,t-1,n,n,1] # i,k
        # oracle similarity

        # similarity_t = (clean_num.unsqueeze(2) * clean_num.unsqueeze(3) + rewards.unsqueeze(2) * rewards.unsqueeze(3)).unsqueeze(-1) # [bs,t-1,n,n,1]
        # similarity_cumsum = th.cumsum(similarity_t,dim=1) # [bs,t-1,n,n,1]
        # similarity = similarity_cumsum.clone()
        # similarity[:,self.sim_horizon:] -= similarity_cumsum[:,:-self.sim_horizon]
        # similarity = similarity/self.sim_horizon

        clean_num_cumsum = th.cumsum(clean_num, dim=1) # [bs,t-1,n]
        rewards_cumsum = th.cumsum(rewards, dim=1) # [bs,t-1,n]
        clean_num_horizon = clean_num_cumsum.clone()
        rewards_horizon = rewards_cumsum.clone()
        clean_num_horizon[:, self.sim_horizon:] -= clean_num_cumsum[:, :-self.sim_horizon]
        rewards_horizon[:, self.sim_horizon:] -= rewards_cumsum[:, :-self.sim_horizon]
        clean_num_t = (clean_num_horizon > 0).float() # {0,1}
        rewards_t = (rewards_horizon > 0).float() # {0,1}

        # x-means
        sample = th.cat([rewards_t.reshape(-1, 1), clean_num_t.reshape(-1, 1)], dim=-1).cpu()
        amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
        xmeans_instance = xmeans(sample, initial_centers, 4)
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()
        which_cluster = th.zeros_like(rewards_t).reshape(-1)
        for i in range(len(clusters)):
            which_cluster[clusters[i]] = i
        which_cluster = which_cluster.reshape_as(rewards_t)
        similarity = (which_cluster.unsqueeze(2) * which_cluster.unsqueeze(3)).unsqueeze(-1)

        # predefined
        if self.env == 'cleanup':
            similarity = (clean_num_t.unsqueeze(2) * clean_num_t.unsqueeze(3) + rewards_t.unsqueeze(2) * rewards_t.unsqueeze(3)).unsqueeze(-1) # [bs,t-1,n,n,1]
        elif self.env == 'harvest':
            similarity = ( rewards_t.unsqueeze(2) * rewards_t.unsqueeze(3) +  (1 - rewards_t.unsqueeze(2)) * (1 - rewards_t.unsqueeze(3))).unsqueeze(-1)
        else:
            similarity = None

        # 4 categories
        # C_H = clean_num_t * rewards_t
        # c_H = (1 - clean_num_t) * rewards_t
        # C_h = clean_num_t * (1 - rewards_t)
        # c_h = (1 - clean_num_t) * (1 - rewards_t)
        #
        #
        # similarity = (C_H.unsqueeze(2) * C_H.unsqueeze(3) +
        #               c_H.unsqueeze(2) * c_H.unsqueeze(3) +
        #               C_h.unsqueeze(2) * C_h.unsqueeze(3) +
        #               c_h.unsqueeze(2) * c_h.unsqueeze(3) ).unsqueeze(-1) # [bs,t-1,n,n,1]


        q_inc_for_sim = th.softmax(q_inc,dim=-1)
        q_of_i_to_j = q_inc_for_sim[:, :-1].unsqueeze(3).repeat(1, 1, 1, self.n_agents, 1, 1)  # [bs,t-1,n,1,n,a_inc]==>[bs,t-1,n,(n),n,a_inc]
        a_of_k_to_j = actions_inc.unsqueeze(2).repeat(1, 1, self.n_agents, 1, 1,1).detach()  # [bs,t-1,1,n,n,1]==>[bs,t-1,(n),n,n,1]
        q_of_i_chosen_by_k_to_j = th.gather(q_of_i_to_j, dim=-1, index=a_of_k_to_j).squeeze( -1)  # [bs,t-1,n,n,n] # of i, k choose, to j, Q
        sim_loss = - th.log(q_of_i_chosen_by_k_to_j) * th.relu(similarity.detach()) * self.env_sim_mask * self.inc_sim_mask
        sim_loss = sim_loss.mean()

        # q_inc_for_sim = q_inc
        # q_of_i_to_j = q_inc_for_sim[:, :-1].unsqueeze(3).repeat(1, 1, 1, self.n_agents, 1, 1)  # [bs,t-1,n,1,n,a_inc]==>[bs,t-1,n,(n),n,a_inc]
        # a_of_k_to_j = actions_inc.unsqueeze(2).repeat(1, 1, self.n_agents, 1, 1, 1).detach()  # [bs,t-1,1,n,n,1]==>[bs,t-1,(n),n,n,1]
        # q_of_i_chosen_by_k_to_j = th.gather(q_of_i_to_j, dim=-1, index=a_of_k_to_j).squeeze(-1)  # [bs,t-1,n,n,n] # of i, k choose, to j, Q
        # sim_loss = (q_of_i_to_j.sum(dim=-1) - 2 * q_of_i_chosen_by_k_to_j) * th.relu(similarity.detach()) * self.env_sim_mask * self.inc_sim_mask
        # sim_loss = sim_loss.mean()
        # sim_loss = th.log(th.exp(sim_loss) + 1.0)

        # ************************************************ step ********************************************************
        self.optimiser_inc.zero_grad()
        self.optimiser_env.zero_grad()
        (value_loss_inc + value_loss_env + sim_loss * self.args.sim_loss_weight).backward(retain_graph=True)
        grad_norm_inc = th.nn.utils.clip_grad_norm_(self.params_inc, self.args.grad_norm_clip)
        grad_norm_env = th.nn.utils.clip_grad_norm_(self.params_env, self.args.grad_norm_clip)
        self.optimiser_inc.step()
        self.optimiser_env.step()

        # self.optimiser_role.zero_grad()
        # loss_identifiable.backward()
        # grad_norm_role = th.nn.utils.clip_grad_norm_(self.params_role, self.args.grad_norm_clip)
        # self.optimiser_role.step()

        # *********************************************** logs *********************************************************

        q_env_taken = th.gather(q_env[:, :-1], dim=-1, index=actions).squeeze(-1)  # [bs,t-1,n]
        q_inc_taken = th.gather(q_inc[:, :-1], dim=-1, index=actions_inc).squeeze(-1)  # [bs,t-1,n,n]

        learning_logs["incentives_to_cleanup"] = (clean_num * receive_value).mean()  # TODOSSD: do we need to log future incentives?
        learning_logs["incentives_to_harvest"] = (rewards * receive_value).mean()
        learning_logs["incentives_to_cleanup_per"] = (clean_num * receive_value).sum()/(clean_num.sum() + 1e-6)
        learning_logs["incentives_to_harvest_per"] = (rewards * receive_value).sum()/(rewards.sum() + 1e-6)

        learning_logs["incentives_zero"] = (actions_inc_masked == 0).float().mean()
        learning_logs["incentives_positive"] = (actions_inc_masked == 1).float().mean()
        learning_logs["incentives_negative"] = (actions_inc_masked == 2).float().mean()

        learning_logs["value_give_mean"] = give_value.float().mean()
        learning_logs["value_give_std"] = give_value.float().std()
        learning_logs["value_receive_mean"] = receive_value.float().mean()
        learning_logs["value_receive_std"] = receive_value.float().std()

        learning_logs["q_env_taken_mean"] = q_env_taken.mean()
        learning_logs["q_inc_taken_mean"] = q_inc_taken.mean()

        # learning_logs['loss_value'] = value_loss # decomposition
        learning_logs['loss_value_env'] = value_loss_env
        learning_logs['loss_value_inc'] = value_loss_inc
        # learning_logs['loss_identifiable'] = loss_identifiable
        # learning_logs['effect_ratio'] = effect_ratio
        learning_logs['loss_sim'] = sim_loss
        learning_logs['similarity_mean'] = similarity.mean()
        learning_logs['similarity_std'] = similarity.std()
        learning_logs['similarity_max'] = similarity.max()
        learning_logs['similarity_min'] = similarity.min()
        cleanup_pair = clean_num.unsqueeze(2) * clean_num.unsqueeze(3) * self.inc_mask
        learning_logs['simlarity_of_cleanup'] = (cleanup_pair.unsqueeze(-1) * th.relu(similarity)).sum() / (cleanup_pair.sum() + 1e-6)
        harvest_pair = rewards.unsqueeze(2) * rewards.unsqueeze(3) * self.inc_mask
        learning_logs['similarity_of_harvest'] = (harvest_pair.unsqueeze(-1) * th.relu(similarity)).sum() / (harvest_pair.sum() + 1e-6)
        # learning_logs['role_mean'] = role.mean()
        # learning_logs['role_std'] = role.std(dim=-1).mean()
        learning_logs['similarity_inc_mean'] = (((actions_inc.unsqueeze(2) - actions_inc.unsqueeze(3))==0).float().mean(dim=(4,5)) * similarity.squeeze(-1)).mean()

        learning_logs['rewards_mean'] = rewards.mean()
        learning_logs['rewards_inc_mean'] = rewards_inc.mean()
        learning_logs['rewards_reassigned_mean'] = rewards_reassigned.mean()

        for i in range(self.n_agents):
            if self.args.env == 'matrix':
                learning_logs['action_env_C_of_{}'.format(i)] = q_env[:, :, i, 0].mean()
                learning_logs["action_env_D_of_{}".format(i)] = q_env[:, :, i, 1].mean()

            learning_logs["actions_inc_R-P_cleanup_of_{}".format(i)] = (
                    (q_inc[:, :-1, i, :, 1] - q_inc[:, :-1, i, :, 2]) * clean_num).mean()
            learning_logs["actions_inc_R-P_harvest_of_{}".format(i)] = (
                    (q_inc[:, :-1, i, :, 1] - q_inc[:, :-1, i, :, 2]) * rewards).mean()

        return learning_logs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1] / self.args.reward_scale  # [bs,t-1,n]
        clean_num = batch["clean_num"][:, :-1]  # [bs,t-1,n]
        # clean_num /= (clean_num.max() + 1e-6)
        apple_den = batch["apple_den"][:, :-1]  # [bs,t-1,n]

        logs = self.cal_loss_and_step(batch)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("clean_num_mean", clean_num.mean().item(), t_env)
            self.logger.log_stat("apple_den_mean", apple_den.mean().item(), t_env)

            # for analysis
            # TODOSSD: more logs, see incentivize_learner for reference

            for k, v in logs.items():
                self.logger.log_stat(k, v.item(), t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        # th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.optimiser_env.state_dict(), "{}/opt_env.th".format(path))
        th.save(self.optimiser_inc.state_dict(), "{}/opt_inc.th".format(path))
        th.save(self.optimiser_role.state_dict(),"{}/opt_role.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # self.optimiser.load_state_dict(
        #     th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_env.load_state_dict(
            th.load("{}/opt_env.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_inc.load_state_dict(
            th.load("{}/opt_inc.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_role.load_state_dict(
            th.load("{}/opt_role.th".format(path), map_location=lambda storage, loc: storage))
