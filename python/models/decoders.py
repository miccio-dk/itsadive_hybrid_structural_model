import torch
from torch.nn import ModuleList
from .layers import IIRNotchTaylor, IIRPeakTaylor, IIRFilterResponse, Multiply, SequentialMultiple

### CUSTOM DECODERS ###

# decoder model using filter design and filtering (1 notch)
class Decoder1N(torch.nn.Module):
    def __init__(self, nfft):
        super().__init__()
        self.iirnotch = IIRNotchTaylor()
        self.iirfilt = IIRFilterResponse(nfft)

    def forward(self, w0, bw):
        b, a = self.iirnotch(w0, bw)
        resp = self.iirfilt(b, a)
        return resp


# decoder model with arbitrary configurable number of parameters
class DecoderCfg(torch.nn.Module):
    def __init__(self, nfft, cfg):
        super().__init__()
        # instantiate layers based on cfg
        layers_list = [self.spectral_feature_block(t, nfft) for t in cfg['spectral_features']]
        self.layers_list = ModuleList(layers_list)
        self.mul = Multiply()

    def spectral_feature_block(self, type, nfft):
        filter = {
            'peak': IIRPeakTaylor,
            'notch': IIRNotchTaylor
        }.get(type)
        return SequentialMultiple(
            filter(),
            IIRFilterResponse(nfft)
        )

    def forward(self, *z):
        # calculate each filter
        resps = [filter_block(*z[i]) for i, filter_block in enumerate(self.layers_list)]
        # total response
        resp = self.mul(resps)
        return resp
