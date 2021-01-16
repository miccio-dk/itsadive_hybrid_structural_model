import torch
from torch.nn import Sequential, ModuleDict

### CUSTOM ENCODERS ###

# encoder model with trainable parameters (1 notch)
class Encoder1N(torch.nn.Module):
    def __init__(self, nfft):
        super().__init__()
        self.lin1 = torch.nn.Linear(nfft // 2 + 1, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.lin_w0 = torch.nn.Linear(16, 1)
        self.lin_bw = torch.nn.Linear(16, 1)

    def forward(self, resp):
        h1_relu = self.lin1(resp).clamp(min=0)
        h2_relu = self.lin2(h1_relu).clamp(min=0)
        w0 = self.lin_w0(h2_relu)
        bw = self.lin_bw(h2_relu)
        return w0, bw


# encoder model with arbitrary configurable number of parameters
class EncoderCfg(torch.nn.Module):
    def __init__(self, nfft, cfg):
        super().__init__()
        params_lookup = {'peak': ['w0', 'bw', 'g'], 'notch': ['w0', 'bw']}
        self.features = {f'{t}{i}': params_lookup[t] for i, t in enumerate(cfg['spectral_features'])}
        input_size = nfft // 2 + 1
        # add first stack
        frontend_sizes = cfg['frontend']['sizes']
        frontend_sizes.insert(0, input_size)
        frontend_act = cfg['frontend']['activation_type']
        frontend_do = cfg['frontend']['dropout_rate']
        self.frontend = self.dense_stack(frontend_sizes, frontend_act, frontend_do)
        # add a stack for each spectral feature to predict (peak or notch)
        specfeat_sizes = cfg['features_block']['sizes']
        specfeat_sizes.insert(0, frontend_sizes[-1])
        specfeat_act = cfg['features_block']['activation_type']
        specfeat_do = cfg['features_block']['dropout_rate']
        specfeat_dict = {k: self.dense_stack(specfeat_sizes, specfeat_act, specfeat_do) for k in self.features}
        self.spectral_features = ModuleDict(specfeat_dict)
        # add a stack for each parameter of each spectral features (w0, bw, g)
        specparams_sizes = cfg['parameters_block']['sizes']
        specparams_sizes.insert(0, specfeat_sizes[-1])
        specparams_sizes.append(1)
        specparams_act = cfg['parameters_block']['activation_type']
        specparams_do = cfg['parameters_block']['dropout_rate']
        specparams_dict = {f'{k}_{p}': self.dense_stack(specparams_sizes, specparams_act, specparams_do) for k in self.features for p in self.features[k]}
        self.spectral_parameters = ModuleDict(specparams_dict)

    def dense_block(self, size_in, size_out, activation_type, dropout_rate):
        activation = {
            'sigmoid': torch.nn.Sigmoid,
            'relu': torch.nn.ReLU,
            'none': torch.nn.Identity
        }.get(activation_type)
        return Sequential(
            torch.nn.Linear(size_in, size_out),
            activation(),
            torch.nn.Dropout(dropout_rate)
        )

    def dense_stack(self, sizes, activation_type, dropout_rate):
        size_args = zip(sizes, sizes[1:])
        layers = [self.dense_block(s_in, s_out, activation_type, dropout_rate) for s_in, s_out in size_args]
        return Sequential(*layers)

    def forward(self, resp):
        z = []
        # run frontend
        x = self.frontend(resp)
        # run spectral feature stacks
        for k in self.features:
            spec_features_x = self.spectral_features[k](x)
            # run spectral parameter stacks
            spec_params_x = [self.spectral_parameters[f'{k}_{p}'](spec_features_x) for p in self.features[k]]
            spec_params_x = tuple(spec_params_x)
            z.append(spec_params_x)
        return tuple(z)
