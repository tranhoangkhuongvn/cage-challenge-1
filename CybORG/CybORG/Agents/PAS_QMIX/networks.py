import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module): 

    def __init__(self, obs, action_space, hidden_1=512, hidden_2=256): 

        super().__init__()
        self.obs = obs
        self.action_space = action_space 
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.feature = nn.Sequential(nn.Linear(obs, hidden_1), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_1, hidden_2), 
                                   nn.ReLU())
        
        self.value_head = nn.Linear(hidden_2, 1)

        self.adv_head = nn.Linear(hidden_2, action_space)

    def forward(self, x):
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = self.feature(x)
        advantage = self.adv_head(x)
        value = self.value_head(x)
		
        result = value + advantage - advantage.mean()
        return result


class ParamNetwork(nn.Module):

    def __init__(self, obs_space, action_space, action_enbedding, hidden_dim=512): 

        super().__init__()
        self.obs_space = obs_space
        self.obs_size = obs_space.shape[0]
        self.action_space = action_space
        self.action_embedding = action_enbedding
        

        self.model = nn.Sequential(nn.Linear(self.obs_size + self.action_embedding, hidden_dim), 
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), 
                                   nn.ReLU())
        #self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(hidden_dim, i) for i in self.action_space])


    def init_hidden(self):
        return torch.zeros((1, self.hidden_dim))


    def forward(self, state_act_concat):
        if not isinstance(state_act_concat, torch.Tensor):
            x = torch.tensor(state_act_concat, dtype=torch.float32)
        else:
            x = state_act_concat
        out = self.model(x)

        #gru_out = self.rnn(out, prev_hidden)
        value = self.value_head(out)

        actions = [head(out) for head in self.adv_heads]
        
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            actions[i] += value

        return actions


class RecurrentQNetwork(nn.Module):

    def __init__(self, observation, action_space, hidden_dim=512):
        super().__init__()
        self.observation = observation
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        # concatenate observation + previous action
        self.fc1 = nn.Linear(observation + action_space, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_space)
        self.ReLU = nn.ReLU()

        self.train()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, obs_action, hidden_state):
        x = self.ReLU(self.fc1(obs_action))
        gru_out = self.rnn(x, hidden_state)
        output = self.fc2(gru_out)
        return output, gru_out


class HyperNetwork(nn.Module):
    
    def __init__(self, state_size, num_agent, qmix_hidden_dim):
        super(HyperNetwork, self).__init__()
        self.state_size = state_size
        self.num_agent = num_agent 
        self.qmix_hidden_dim = qmix_hidden_dim
        
        self.w1_layer = nn.Linear(self.state_size, self.num_agent * self.qmix_hidden_dim)
        self.w2_layer = nn.Linear(self.state_size, self.qmix_hidden_dim)
        self.b1_layer = nn.Linear(self.state_size, self.qmix_hidden_dim)
        self.b2_layer = nn.Linear(self.state_size, 1)
        
        self.LReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()

        #self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer.weight)
        

    def forward(self, state):
        w1_shape = self.hyper_net_pars['w1_shape']
        w2_shape = self.hyper_net_pars['w2_shape']
        w1 = torch.abs(self.w1_layer(state)).view(-1, w1_shape[0], w1_shape[1])
        w2 = torch.abs(self.w2_layer(state)).view(-1, w2_shape[0], w2_shape[1])
        b1 = self.b1_layer(state).view(-1, 1, self.hyper_net_pars['b1_shape'][0])
        #x = self.LReLU(self.b2_layer_i(state))
        x = self.ReLU(self.b2_layer_i(state))
        b2 = self.b2_layer_h(x).view(-1, 1, self.hyper_net_pars['b2_shape'][0])
        return {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
        

class Mixing_Network(nn.Module):
    def __init__(self, action_size, num_agents, args):
        super(Mixing_Network, self).__init__()
        # action_size * num_agents = the num of Q values
        self.w1_shape = torch.Size((num_agents, args.mix_net_out[0]))
        self.b1_shape = torch.Size((args.mix_net_out[0], ))
        self.w2_shape = torch.Size((args.mix_net_out[0], args.mix_net_out[1]))
        self.b2_shape = torch.Size((args.mix_net_out[1], ))
        self.w1_size = self.w1_shape[0] * self.w1_shape[1]
        self.b1_size = self.b1_shape[0]
        self.w2_size = self.w2_shape[0] * self.w2_shape[1]
        self.b2_size = self.b2_shape[0]
        self.pars = {'w1_shape':self.w1_shape, 'w1_size':self.w1_size, \
                'w2_shape':self.w2_shape, 'w2_size':self.w2_size, \
                'b1_shape':self.b1_shape, 'b1_size':self.b1_size, \
                'b2_shape':self.b2_shape, 'b2_size':self.b2_size, }
        self.LReLU = nn.LeakyReLU(0.001)
        self.ReLU = nn.ReLU()
    
    def forward(self, q_values, hyper_pars):
        x = self.ReLU(torch.bmm(q_values, hyper_pars['w1']) + hyper_pars['b1'])
        output = torch.bmm(x, hyper_pars['w2']) + hyper_pars['b2']
        return output.view(-1)