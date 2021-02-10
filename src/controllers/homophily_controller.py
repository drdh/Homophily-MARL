from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# This multi-agent controller shares parameters between agents
class HomophilyMAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super(HomophilyMAC, self).__init__()
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.h_env = None
        self.h_inc = None

        self.extra_return_env = None
        self.extra_return_inc = None

        self.inc_mask_actions = (1 - th.eye(self.n_agents)).reshape(1, self.n_agents, self.n_agents, 1).to(self.args.device)

    def select_actions_env(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_env = self.forward_env(ep_batch, t_ep, test_mode=test_mode)
        masks = avail_actions[bs] # [bs,n,n_env]
        chosen_actions = self.action_selector.select_action(q_env[bs], masks , t_env, test_mode=test_mode)
        chosen_actions = chosen_actions.unsqueeze(-1)

        return chosen_actions
        # [bs,n,1]

    def select_actions_inc(self, chosen_actions, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, agent_pos_replay = None):
        q_inc = self.forward_inc(ep_batch, t_ep, chosen_actions, test_mode=test_mode)
        rewards = ep_batch["reward"][:,t_ep] # [bs,n]
        masks = th.ones_like(q_inc[bs]) # [bs,n,n,a_inc]
        chosen_actions_inc = self.action_selector.select_action(q_inc[bs], masks, t_env,test_mode=test_mode)
        chosen_actions_inc = chosen_actions_inc.unsqueeze(-1) * self.inc_mask_actions # # [bs,n,n,1]

        if self.args.save_replay == True:
            if t_ep>0:
                data = agent_pos_replay
                plt.clf()  #

                incentives = chosen_actions_inc.squeeze(0).squeeze(-1)  # [n,n]
                from_dev = 0.2
                to_dev = -0.2
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        if i != j and incentives[i, j] != 0:
                            color = 'lime' if incentives[i, j] == 1 else 'deepskyblue'
                            plt.arrow(x=data[i, 1] + from_dev,
                                      y=data[i, 0] + from_dev,
                                      dx=data[j, 1] - data[i, 1] + to_dev,
                                      dy=data[j, 0] - data[i, 0] + to_dev,
                                      alpha=0.8, width=0.1, head_width=0.8, color=color)
        return chosen_actions_inc
                # [bs,n,n,1]

    def forward_env(self, ep_batch, t, test_mode=False, learning_mode=False):
        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,-1]
        q_env, self.h_env, self.extra_return_env = self.agent.forward_env(self.agent_inputs, self.h_env, learning_mode)  # [bs,n_a]
        return q_env.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_inc(self, ep_batch, t, chosen_actions, test_mode=False, learning_mode=False):
        actions_env = F.one_hot(chosen_actions.squeeze(-1), num_classes=self.args.n_actions)

        q_inc, self.h_inc, self.extra_return_inc = self.agent.forward_inc(self.agent_inputs, self.h_inc,
                                    actions_env,
                                    ep_batch["agent_pos"][:, t] / np.linalg.norm(self.args.state_dims),
                                    ep_batch["agent_orientation"][:, t],
                                    ep_batch["reward"][:, t].unsqueeze(-1),  # [bs,n,1] #
                                    ep_batch["clean_num"][:, t].unsqueeze(-1),  # [bs,n,1]
                                    ep_batch["apple_den"][:, t].unsqueeze(-1),  # [bs,n,1]
                                    learning_mode=learning_mode,
                                    )  # [bs,n,n,a_inc]

        return q_inc.view(ep_batch.batch_size, self.n_agents, self.n_agents, -1)

    # only for learning
    def forward(self, ep_batch, t, test_mode=False):
        q_env = self.forward_env(ep_batch, t, test_mode=test_mode, learning_mode=True)
        chosen_actions = ep_batch["actions"][:, t, :]
        q_inc = self.forward_inc(ep_batch, t, chosen_actions, test_mode=test_mode, learning_mode=True)
        return q_env, q_inc, self.extra_return_inc
        # [bs,n,a_env] [bs,n,n,a_inc]

    def init_hidden(self, batch_size):
        self.h_env,self.h_inc = self.agent.init_hidden()
        self.h_env = self.h_env.repeat(batch_size,1,1,1)
        self.h_inc = self.h_inc.repeat(batch_size,1,1,1)

    # parameters for different training settings
    def parameters(self):
        return self.agent.parameters()

    def parameters_env(self):
        return self.agent.parameters_env()

    def parameters_inc(self):
        return self.agent.parameters_inc()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        if self.args.rgb_input:
            data = batch['obs'][:, t]
            data = self.agent.rgb_preprocess(
                data.reshape((bs * self.n_agents, 3, self.args.obs_dims[0], self.args.obs_dims[1])))  # [bs*n,...]
            inputs.append(data.reshape((bs, self.n_agents, self.args.obs_dim_net)))  # [bs,n,...]
        else:
            inputs.append(batch["obs"][:, t])  # b1av, [bs,t,n,..] ==> [bs,n,...]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.args.obs_reward:
            if t == 0:
                inputs.append(th.zeros_like(batch["reward"][:, t]))
            else:
                rewards = batch["reward"][:, t - 1].clone()  # [bs,n]
                inputs.append(th.sign(rewards))

        if self.args.obs_inc_reward:
            if t == 0:
                inputs.append(th.zeros_like(batch["reward"][:, t]))
            else:
                actions_inc = batch["actions_inc"][:, t - 1]  # [bs,n,n,1]
                actions_inc_masked = actions_inc * self.inc_mask_actions
                receive_value = th.stack([
                    th.sum(actions_inc_masked[:, :, i] == 1, dim=(1, 2)) \
                    - th.sum(actions_inc_masked[:, :, i] == 2, dim=(1, 2))
                    for i in range(self.n_agents)], dim=-1)
                inputs.append(th.sign(receive_value.float()))

        if self.args.obs_others_last_action:
            bs = batch["actions_onehot"][:, t].shape[0]
            if t == 0:
                inputs.append(th.zeros_like(
                    batch["actions_onehot"][:, t].repeat(1, self.n_agents, 1).reshape(bs, self.n_agents, -1)))
            else:
                inputs.append(
                    batch["actions_onehot"][:, t - 1].repeat(1, self.n_agents, 1).reshape(bs, self.n_agents, -1))
        if self.args.obs_distance:
            agent_pos = batch["agent_pos"][:, t]  # [bs,n,2]
            agent_distance = 1.0 - (agent_pos.unsqueeze(2) - agent_pos.unsqueeze(1)).norm(dim=-1) / (
                np.linalg.norm(self.args.state_dims))  # [bs,n,n]
            inputs.append(agent_distance)
        if self.args.obs_agent_pos:
            agent_pos = batch["agent_pos"][:, t] / np.linalg.norm(self.args.state_dims)  # [bs,n,2]
            inputs.append(agent_pos)

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        if self.args.rgb_input:
            input_shape = self.args.obs_dim_net
        else:
            input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_reward:
            input_shape += 1
        if self.args.obs_inc_reward:
            input_shape += 1
        if self.args.obs_others_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        if self.args.obs_distance:
            input_shape += self.n_agents
        if self.args.obs_agent_pos:
            input_shape += 2

        return input_shape
