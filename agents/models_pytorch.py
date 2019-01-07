import torch
import torch.nn as nn
import pdb 

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    

class dqn_model(nn.Module):
        
    def __init__(self, observation_space, conv_params, hidden_params, layer_norm, activation_fn, n_actions):
        super(dqn_model, self).__init__()
        
        self.observation_space = observation_space        
        self.conv = []
        self.fc_action = [] 
        self.fc_state = []
        self.normalize_action = []
        self.normalize_state = []
#        
        conv = []
        size_x = observation_space[0]
        size_y = observation_space[1]
        size_z = observation_space[2]
        
        for i in range(conv_params.shape[0]):
            if i == 0:
                in_channel = size_z
            else:
                in_channel = conv_params[i-1][0]
                
            out_channel = conv_params[i][0]
            kernel_size = conv_params[i][1]
            stride = conv_params[i][2]
            padding = conv_params[i][3]
            
            size_x = int((size_x - kernel_size + 2*padding)/stride + 1)
            size_y = int((size_y - kernel_size + 2*padding)/stride + 1)
            
            conv.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))              
        self.conv = ListModule(*conv)
        
        fc_action = []
        normalize_action = []        
        fc_state = []
        normalize_state = []
        
        for i in range(hidden_params.shape[0]):                                
            if i == 0:
                in_channel = size_x * size_y * conv_params[-1][0]
            else:
                in_channel = hidden_params[i - 1]
                 
            fc_action.append(nn.Linear(in_channel, hidden_params[i]))
            fc_state.append(nn.Linear(in_channel, hidden_params[i]))            
            
            if layer_norm:                
                normalize_action.append(nn.LayerNorm(fc_action[-1].out_features))
                normalize_state.append(nn.LayerNorm(fc_state[-1].out_features))
                                    
        fc_action.append(nn.Linear(hidden_params[-1], n_actions))
        fc_state.append(nn.Linear(hidden_params[-1], 1))
        
        self.fc_action = ListModule(*fc_action)
        self.fc_state = ListModule(*fc_state)
        self.normalize_action = ListModule(*normalize_action)
        self.normalize_state = ListModule(*normalize_state)
            
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm       
        
        for params in self.parameters():
            params.retain_grad()

    def forward(self, x):                
        x = x/255.0
        for i in range(len(self.conv)):
            x = self.activation_fn(self.conv[i](x))
            
        encoder_out_shape = x.shape
        
        x = x.view(x.shape[0], -1)
        
        middle_out1 = x
        middle_out2 = x.clone()
        
        for i in range(len(self.fc_action) - 1):
            
            middle_out1 = self.fc_action[i](middle_out1)
            middle_out1 = self.normalize_action[i](middle_out1)        
            action_scores = self.activation_fn(middle_out1)
            
            middle_out2 = self.fc_state[i](middle_out2)
            middle_out2 = self.normalize_state[i](middle_out2)        
            state_scores = self.activation_fn(middle_out2)

        action_scores = self.fc_action[len(self.fc_action) - 1](action_scores) # No activation function for last layer
        state_scores = self.fc_state[len(self.fc_action) - 1](state_scores) # No activation function for last layer
        
        output = state_scores + action_scores - torch.mean(action_scores) 
        
        return output        
    
class qmap_model(nn.Module):

    def __init__(self, observation_space, conv_params, hidden_params, deconv_params, layer_norm, activation_fn):
        super(qmap_model, self).__init__()
        
        self.conv = []
        self.fc = [] 
        self.deconv_action = []
        self.deconv_state = []
        self.normalize = []    
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm      
        
        size_x = observation_space[0]
        size_y = observation_space[1]
        size_z = observation_space[2]
        
        conv = []
        for i in range(conv_params.shape[0]):
            if i == 0:
                in_channel = size_z                
            else:
                in_channel = conv_params[i-1][0]
                
            out_channel = conv_params[i][0]
            kernel_size = conv_params[i][1]            
            stride = conv_params[i][2]
            padding = conv_params[i][3]
            
            size_x = int((size_x - kernel_size + 2*padding)/stride + 1)
            size_y = int((size_y - kernel_size + 2*padding)/stride + 1)
            
            conv.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        self.conv = ListModule(*conv)
        
        fc = []
        normalize = []
        
        for i in range(hidden_params.shape[0]):                    
            if i == 0:
                in_channel = size_x * size_y * conv_params[-1][0]
            else:
                in_channel = hidden_params[i - 1]
            
            fc.append(nn.Linear(in_channel, hidden_params[i]))
            if layer_norm:
                normalize.append(nn.LayerNorm(fc[-1].out_features))
        
        fc.append(nn.Linear(hidden_params[i], size_x * size_y * conv_params[-1][0]))
        if layer_norm:
                normalize.append(nn.LayerNorm(fc[-1].out_features))
        
        self.fc = ListModule(*fc)
        self.normalize = ListModule(*normalize)
            
        deconv_action = []
        deconv_state = []        
        
        for i in range(deconv_params.shape[0]):            
            if i == 0:
                in_channel = conv_params[-1][0]
            else:
                in_channel = deconv_params[i-1][0]
            
            out_channel = deconv_params[i][0]
            kernel_size = deconv_params[i][1]
            stride = deconv_params[i][2] 
            padding = deconv_params[i][3]            
               
            deconv_action.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding))
        
            if i == deconv_params.shape[0] - 1: # Only one State value finally
                out_channel = 1
                
            deconv_state.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding))
            
        self.deconv_action = ListModule(*deconv_action)
        self.deconv_state = ListModule(*deconv_state)        
        
        for params in self.parameters():
            params.retain_grad()

    def forward(self, x):                 
        x = x/255.0
        for i in range(len(self.conv)):
            x = self.activation_fn(self.conv[i](x))
            
        encoder_out_shape = x.shape                
        x = x.view(x.shape[0], -1)
        
        for i in range(len(self.fc)):
            if self.layer_norm:
                x = self.activation_fn(self.normalize[i](self.fc[i](x)))
            else:
                x = self.activation_fn(self.fc[i](x))
        
        x = x.view(encoder_out_shape)
        middle_out1 = x
        middle_out2 = x.clone()        
        
        for i in range(len(self.deconv_action) - 1):            
            middle_out1 = self.activation_fn(self.deconv_action[i](middle_out1))
        
        middle_out1 = nn.ZeroPad2d((0, 1, 0, 1))(middle_out1)
        action_scores = self.deconv_action[len(self.deconv_action) - 1](middle_out1) # No activation function for last layer
        
        for i in range(len(self.deconv_state) - 1):
            middle_out2 = self.activation_fn(self.deconv_state[i](middle_out2))

        middle_out2 = nn.ZeroPad2d((0, 1, 0, 1))(middle_out2)
        state_scores = self.deconv_state[len(self.deconv_action) - 1](middle_out2) # No activation function for last layer        
                
        output = state_scores + action_scores - torch.mean(action_scores, 1).unsqueeze(1)         
        return output


