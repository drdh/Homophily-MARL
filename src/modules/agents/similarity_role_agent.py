import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from itertools import chain
import math


# from utils.helper import Flatten

class SimilarityRoleAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SimilarityRoleAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_inc_actoins = args.n_inc_actions
        self.input_shape = input_shape
        self.extra_input_shape = args.n_actions + 2 + 2 + 3  # pos,orientation,[reward,clean_num,apple_den]

        self.n_env_networks = self.n_agents
        self.n_role_networks = 1
        self.role_dim = args.role_dim
        self.nn_hidden_size = 16

        if args.rgb_input:
            self.conv_to_fc = nn.Sequential(
                nn.Conv2d(3, args.conv_out, args.conv_kernel, args.conv_stride),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(args.conv_out * (args.obs_dims[0] - args.conv_kernel + 1) * (
                        args.obs_dims[1] - args.conv_kernel + 1), args.obs_dim_net),
                nn.LeakyReLU()
            )

        def init_w(tensor):
            return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

        def init_b(tensor, fan_in):
            bound = 1 / math.sqrt(fan_in)
            return nn.init.uniform_(tensor, -bound, bound)

        # *********************************************** env **********************************************************#
        self.fc1_env_w = nn.Parameter(init_w(th.Tensor(1, self.n_env_networks, input_shape, args.rnn_hidden_dim)))
        self.fc1_env_b = nn.Parameter(init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), input_shape), )

        # rnn begin
        self.rnn_env_ir_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_ir_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_env_hr_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_hr_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_env_iz_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_iz_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_env_hz_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_hz_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_env_in_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_in_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_env_hn_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_env_hn_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_env_networks, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))
        # rnn end

        # fc2
        self.fc2_env_w = nn.Parameter(init_w(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, self.n_actions)))
        self.fc2_env_b = nn.Parameter(init_b(th.Tensor(1, self.n_env_networks, 1, self.n_actions), args.rnn_hidden_dim))

        self.fc2_env_v_w = nn.Parameter(init_w(th.Tensor(1, self.n_env_networks, args.rnn_hidden_dim, 1)))
        self.fc2_env_v_b = nn.Parameter(init_b(th.Tensor(1, self.n_env_networks, 1, 1), args.rnn_hidden_dim))

        # **************************************************** role ***************************************************#

        # role encoder
        self.role_enc_fc1_w = nn.Parameter(init_w(th.Tensor(1, self.n_role_networks, self.extra_input_shape, self.nn_hidden_size)))
        self.role_enc_fc1_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_role_networks, 1, self.nn_hidden_size), input_shape))
        self.role_enc_fc2_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_role_networks, self.nn_hidden_size, self.role_dim * 2)))
        self.role_enc_fc2_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_role_networks, 1, self.role_dim * 2), self.nn_hidden_size))

        # self.role_enc_fc = nn.Parameter(init_w(th.Tensor(1, self.n_role_networks, self.extra_input_shape, self.role_dim * 2)))

        # trajectory encoder
        self.traj_enc_fc1_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_role_networks, args.rnn_hidden_dim + self.extra_input_shape, self.nn_hidden_size)))
        self.traj_enc_fc1_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_role_networks, 1, self.nn_hidden_size), args.rnn_hidden_dim + input_shape))
        self.traj_enc_fc2_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_role_networks, self.nn_hidden_size, self.role_dim * 2)))
        self.traj_enc_fc2_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_role_networks, 1, self.role_dim * 2), self.nn_hidden_size))



        # ************************************************* inc *******************************************************#
        self.fc1_inc_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_agents, input_shape + self.n_actions, args.rnn_hidden_dim)), )
        self.fc1_inc_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), input_shape + self.n_actions), )

        # rnn begin
        self.rnn_inc_ir_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_ir_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_inc_hr_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_hr_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_inc_iz_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_iz_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_inc_hz_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_hz_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_inc_in_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_in_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))

        self.rnn_inc_hn_w = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim), args.rnn_hidden_dim))
        self.rnn_inc_hn_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.rnn_hidden_dim), args.rnn_hidden_dim))
        # rnn end

        self.fc2_inc_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_agents, args.rnn_hidden_dim + self.extra_input_shape, args.n_inc_actions)))
        self.fc2_inc_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, args.n_inc_actions), args.rnn_hidden_dim + self.extra_input_shape))

        self.fc2_inc_v_w = nn.Parameter(
            init_w(th.Tensor(1, self.n_agents, args.rnn_hidden_dim + self.extra_input_shape, 1)))
        self.fc2_inc_v_b = nn.Parameter(
            init_b(th.Tensor(1, self.n_agents, 1, 1), args.rnn_hidden_dim + self.extra_input_shape))

    def parameters_env(self):
        params = []
        if self.args.rgb_input:
            params += list(self.conv_to_fc.parameters())

        for n, p in self.named_parameters():
            if 'env' in n:
                params.append(p)

        return params

    def parameters_inc(self):
        params = []
        if self.args.rgb_input:
            params += list(self.conv_to_fc.parameters())

        for n, p in self.named_parameters():
            if 'inc' in n:
                params.append(p)
        return params

    def parameters_role(self):
        params = []

        for n, p in self.named_parameters():
            if 'enc' in n:
                params.append(p)

        return params

    def init_hidden(self):
        h_env = self.fc1_env_w.new_zeros(1, self.n_agents, 1, self.args.rnn_hidden_dim).detach()
        h_inc = self.fc1_inc_w.new_zeros(1, self.n_agents, 1, self.args.rnn_hidden_dim).detach()
        return h_env, h_inc

    def forward_env(self, inputs, h_in, learning_mode=False):
        inputs = inputs.reshape(-1, self.n_agents, 1, self.input_shape)  # [bs,n,1,obs]
        h_in = h_in.reshape(-1, self.n_agents, 1, self.args.rnn_hidden_dim)  # [bs,n,1,rnn]

        # fc1
        x = F.leaky_relu(th.matmul(inputs, self.fc1_env_w) + self.fc1_env_b)  # [bs,n,1,rnn]

        # rnn
        r = th.sigmoid(th.matmul(x, self.rnn_env_ir_w) + self.rnn_env_ir_b + th.matmul(h_in,self.rnn_env_hr_w) + self.rnn_env_hr_b)  # [bs,n,1,rnn]
        z = th.sigmoid(th.matmul(x, self.rnn_env_iz_w) + self.rnn_env_iz_b + th.matmul(h_in,self.rnn_env_hz_w) + self.rnn_env_hz_b)  # [bs,n,1,rnn]
        n = th.tanh(th.matmul(x, self.rnn_env_in_w) + self.rnn_env_in_b + r * (th.matmul(h_in, self.rnn_env_hn_w) + self.rnn_env_hn_b))  # [bs,n,1,rnn]
        h_out = (1 - z) * n + z * h_in  # [bs,n,1,rnn]

        # all
        a_env = th.matmul(h_out, self.fc2_env_w) + self.fc2_env_b  # [bs,n,1,a_env]
        v_env = th.matmul(h_out, self.fc2_env_v_w) + self.fc2_env_v_b  # [bs,n,1,1]
        q_env = v_env + a_env - a_env.mean(dim=-1, keepdim=True)

        extra_return = {}
        return q_env.squeeze(2), h_out, extra_return
        # [bs,n,a_env], [bs,n,1,rnn]

        # [bs,n,a_env], [bs,n,rnn_dim] [bs,n,2] [bs,n,2]

    def forward_inc(self, inputs, h_in, actions_env, agent_pos, agent_orientation, reward, clean_num, apple_den, learning_mode = False):
        inputs_inc = inputs.reshape(-1, self.n_agents, 1, self.input_shape)  # [bs,n,1,obs]
        actions_env = actions_env.type_as(inputs_inc)
        h_in = h_in.reshape(-1, self.n_agents, 1, self.args.rnn_hidden_dim)  # [bs,n,1,rnn]
        actions_env = actions_env.reshape(-1, self.n_agents, 1, self.n_actions)

        extra_return = {}
        # if learning_mode:
        #     # role inputs
        #     agent_pos = agent_pos.reshape(-1, self.n_agents, 1, 2)
        #     agent_orientation = agent_orientation.reshape(-1, self.n_agents, 1, 2)
        #     reward = reward.reshape(-1, self.n_agents, 1, 1)
        #     clean_num = clean_num.reshape(-1, self.n_agents, 1, 1)
        #     apple_den = apple_den.reshape(-1, self.n_agents, 1, 1)
        #
        #     inputs_role = th.cat([actions_env,agent_pos,agent_orientation,reward,clean_num,apple_den],dim=-1) # [bs,n,1,extra]
        #
        #     # role encoder
        #     enc1 = F.leaky_relu(th.matmul(inputs_role.detach(), self.role_enc_fc1_w) + self.role_enc_fc1_b)  # [bs,n,1,nn_h]
        #     role_param = th.matmul(enc1, self.role_enc_fc2_w) + self.role_enc_fc2_b  # [bs,n,1,role*2]
        #     # role_param = th.matmul(inputs.detach(), self.role_enc_fc) # [bs,n,1,role*2]
        #     role_gaussian = D.Normal(role_param[:, :, :, :self.role_dim], th.exp(role_param[:, :, :, self.role_dim:]))
        #     role = role_gaussian.rsample()  # [bs,n,1,role]
        #
        #     extra_return = {
        #         'role': role.squeeze(2),  # [bs,n,role]
        #     }
        #
        #     traj1 = F.leaky_relu(th.matmul(th.cat([h_in.detach(),inputs_role.detach()],dim=-1), self.traj_enc_fc1_w) + self.traj_enc_fc1_b) # [bs,n,1,nn_h]
        #     role_infer_param = th.matmul(traj1,self.traj_enc_fc2_w) + self.traj_enc_fc2_b # [bs,n,1,role*2]
        #     role_infer_gaussian = D.Normal(role_infer_param[:,:,:,:self.role_dim], th.exp(role_infer_param[:,:,:,self.role_dim:]))
        #     loss_identifiable = D.kl_divergence(role_gaussian, role_infer_gaussian).mean() # TODOSSD: role_gaussian.entropy().mean()?
        #     extra_return['loss_identifiable'] = loss_identifiable


        # inc
        x = F.leaky_relu(th.matmul(th.cat([inputs_inc,actions_env],dim=-1), self.fc1_inc_w) + self.fc1_inc_b)  # [bs,n,1,rnn]
        r = th.sigmoid(th.matmul(x, self.rnn_inc_ir_w) + self.rnn_inc_ir_b + th.matmul(h_in, self.rnn_inc_hr_w) + self.rnn_inc_hr_b)  # [bs,n,1,rnn]
        z = th.sigmoid(th.matmul(x, self.rnn_inc_iz_w) + self.rnn_inc_iz_b + th.matmul(h_in,self.rnn_inc_hz_w) + self.rnn_inc_hz_b)  # [bs,n,1,rnn]
        n = th.tanh(th.matmul(x, self.rnn_inc_in_w) + self.rnn_inc_in_b + r * ( th.matmul(h_in, self.rnn_inc_hn_w) + self.rnn_inc_hn_b))  # [bs,n,1,rnn]
        h_out = (1 - z) * n + z * h_in  # [bs,n,1,rnn]

        # inc, inputs
        actions_env = actions_env.reshape(-1, 1, self.n_agents, self.n_actions).repeat(1, self.n_agents, 1,1)  # [bs,(n),n,a_env]
        agent_pos = agent_pos.reshape(-1, 1, self.n_agents, 2).repeat(1, self.n_agents, 1, 1)  # [bs,(n),n,2]
        agent_orientation = agent_orientation.reshape(-1, 1, self.n_agents, 2).repeat(1, self.n_agents, 1, 1)  # [bs,(n),n,2]
        reward = reward.reshape(-1, 1, self.n_agents, 1).repeat(1, self.n_agents, 1, 1)  # [bs,(n),n,1]
        clean_num = clean_num.reshape(-1, 1, self.n_agents, 1).repeat(1, self.n_agents, 1, 1)  # [bs,(n),n,1]
        apple_den = apple_den.reshape(-1, 1, self.n_agents, 1).repeat(1, self.n_agents, 1, 1)  # [bs,(n),n,1]
        inputs_cat = th.cat([h_out.repeat(1,1,self.n_agents,1), actions_env, agent_pos, agent_orientation, reward, clean_num, apple_den],
                            dim=-1)  # [bs,n,n,dim], i to j

        # all
        a_inc = th.matmul(inputs_cat, self.fc2_inc_w) + self.fc2_inc_b  # [bs,n,n,a_inc]
        v_inc = th.matmul(inputs_cat, self.fc2_inc_v_w) + self.fc2_inc_v_b  # [bs,n,n,1]
        q_inc = v_inc + a_inc - a_inc.mean(dim=-1, keepdim=True)

        return q_inc, h_out, extra_return
        # [bs,n,n,a_inc], [bs,n,n,rnn]  # {n}->{n}

        # [bs*n,obs] # [bs*n,n,rnn]

    def rgb_preprocess(self, x):
        return self.conv_to_fc(x)
