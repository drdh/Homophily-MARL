import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer



class HomophilyLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.inc_mask = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, self.n_agents).to(self.device)
        self.inc_mask_actions = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, self.n_agents, 1).to(self.device)

        self.env_sim_mask = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, self.n_agents, 1).to(self.device)
        self.inc_sim_mask = (1 - th.eye(self.n_agents)).reshape(1, 1, self.n_agents, 1, self.n_agents).to(self.device)
        self.oth_sim_mask = (1 - th.eye(self.n_agents)).reshape(1, 1, 1, self.n_agents, self.n_agents).to(self.device)
        # [bs,t-1,n,n,n] # of i, k choose, to j, Q


        self.inc_scale = 1  # self.n_agents
        self.inc_cost_scale = 1  # /(self.n_agents - 1)

        self.sim_horizon = args.sim_horizon

        self.env = args.env

        self.params = list(mac.parameters())
        self.params_env = mac.parameters_env()
        self.params_inc = mac.parameters_inc()

        self.last_target_update_episode = 0

        self.optimiser_env = Adam(params=self.params_env, lr=args.lr_env)
        self.optimiser_inc = Adam(params=self.params_inc, lr=args.lr_inc)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def cal_loss_and_step(self, batch: EpisodeBatch):
        learning_logs = {}
        # ********************************************* data from buffer ***********************************************
        rewards = batch["reward"][:, :-1] / self.args.reward_scale  # [bs,t-1,n]
        actions = batch["actions"][:, :-1]  # [bs,t-1,n,1]
        actions_inc = batch["actions_inc"][:, :-1]  # [bs,t-1,n,n,1]
        actions_inc_all = batch["actions_inc"]  # [bs,t,n,n,1]

        clean_num = batch["clean_num"][:, :-1].clone()  # [bs,t-1,n]
        clean_num = (clean_num > 0).float()

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # [bs,t-1,1]
        avail_actions = batch["avail_actions"]

        # ************************************************ mac out *****************************************************
        q_env = []
        q_inc = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q_env_t, q_inc_t, extra_return = self.mac.forward(batch, t=t)
            q_env.append(q_env_t)  # [bs,n,a_env]
            q_inc.append(q_inc_t)  # [bs,n,n,a_inc]
        q_env = th.stack(q_env, dim=1)  # [bs,t,n,a_env]
        q_inc = th.stack(q_inc, dim=1)  # [bs,t,n,n,a_inc]

        avail_inc_actions = th.ones_like(q_inc) # [bs,t,n,n,a_inc]

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
        is_idle = clean_num_t + rewards_t
        idle_agent = (is_idle.unsqueeze(2) * is_idle.unsqueeze(3)).unsqueeze(-1)  # [bs,t-1,n,n,1]
        similarity = (which_cluster.unsqueeze(2) == which_cluster.unsqueeze(3)).unsqueeze(-1).float() * idle_agent


        q_inc_for_sim = th.softmax(q_inc,dim=-1)
        q_of_i_to_j = q_inc_for_sim[:, :-1].unsqueeze(3).repeat(1, 1, 1, self.n_agents, 1, 1)  # [bs,t-1,n,1,n,a_inc]==>[bs,t-1,n,(n),n,a_inc]
        a_of_k_to_j = actions_inc.unsqueeze(2).repeat(1, 1, self.n_agents, 1, 1,1).detach()  # [bs,t-1,1,n,n,1]==>[bs,t-1,(n),n,n,1]
        q_of_i_chosen_by_k_to_j = th.gather(q_of_i_to_j, dim=-1, index=a_of_k_to_j).squeeze( -1)  # [bs,t-1,n,n,n] # of i, k choose, to j, Q

        sim_loss_mask = th.relu(similarity.detach()) * self.env_sim_mask * self.inc_sim_mask * self.oth_sim_mask
        sim_loss = - th.log(q_of_i_chosen_by_k_to_j)
        sim_loss = th.clamp_min(sim_loss, self.args.sim_threshold) * sim_loss_mask
        sim_loss = sim_loss.sum() / (1 + sim_loss_mask.sum())

        # ************************************************ step ********************************************************
        self.optimiser_inc.zero_grad()
        self.optimiser_env.zero_grad()
        (value_loss_inc + value_loss_env + sim_loss * self.args.sim_loss_weight).backward(retain_graph=True)
        grad_norm_inc = th.nn.utils.clip_grad_norm_(self.params_inc, self.args.grad_norm_clip)
        grad_norm_env = th.nn.utils.clip_grad_norm_(self.params_env, self.args.grad_norm_clip)
        self.optimiser_inc.step()
        self.optimiser_env.step()


        # *********************************************** logs *********************************************************

        q_env_taken = th.gather(q_env[:, :-1], dim=-1, index=actions).squeeze(-1)  # [bs,t-1,n]
        q_inc_taken = th.gather(q_inc[:, :-1], dim=-1, index=actions_inc).squeeze(-1)  # [bs,t-1,n,n]

        learning_logs["incentives_to_cleanup_per"] = (clean_num * receive_value).sum()/(clean_num.sum() + 1e-6)
        learning_logs["incentives_to_harvest_per"] = (rewards * receive_value).sum()/(rewards.sum() + 1e-6)

        learning_logs["value_give_mean"] = give_value.float().mean()
        learning_logs["value_receive_mean"] = receive_value.float().mean()

        learning_logs["q_env_taken_mean"] = q_env_taken.mean()
        learning_logs["q_inc_taken_mean"] = q_inc_taken.mean()

        learning_logs['loss_value_env'] = value_loss_env
        learning_logs['loss_value_inc'] = value_loss_inc
        learning_logs['loss_sim'] = sim_loss

        return learning_logs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        clean_num = batch["clean_num"][:, :-1]  # [bs,t-1,n]
        apple_den = batch["apple_den"][:, :-1]  # [bs,t-1,n]

        logs = self.cal_loss_and_step(batch)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("clean_num_mean", clean_num.mean().item(), t_env)
            self.logger.log_stat("apple_den_mean", apple_den.mean().item(), t_env)

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
        th.save(self.optimiser_env.state_dict(), "{}/opt_env.th".format(path))
        th.save(self.optimiser_inc.state_dict(), "{}/opt_inc.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser_env.load_state_dict(
            th.load("{}/opt_env.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_inc.load_state_dict(
            th.load("{}/opt_inc.th".format(path), map_location=lambda storage, loc: storage))

