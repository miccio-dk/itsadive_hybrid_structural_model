import numpy as np
import torch
from torch.fft import rfft
from scipy.interpolate import approximate_taylor_polynomial

# calculate polynomial
def polyval(coeffs, x):
    curVal = torch.zeros(x.shape)
    for curValIndex in range(len(coeffs) - 1):
        curVal = (curVal + coeffs[curValIndex]) * x
    return (curVal + coeffs[len(coeffs) - 1])

### CUSTOM LAYERS ###

# iir notch filter design layer (with taylor approximation)
class IIRNotchTaylor(torch.nn.Module):
    def __init__(self, order=5, scale=0.5):
        super(IIRNotchTaylor, self).__init__()
        self.x0 = 0.5
        self.p_coeffs = self._get_polynomial(order, scale)

    def _get_polynomial(self, order, scale):
        # this is the function for the gain, given a normalized bandwidth
        def f(x):
            return 1 / (1 + np.tan(x * np.pi / 2))
        # calculate taylor series coefficients
        p = approximate_taylor_polynomial(f, x=self.x0, degree=order, scale=scale)
        return p.coeffs.astype('float32')

    def forward(self, w0, bw):
        w0 = w0 * np.pi
        # use taylor-series approximation for gain calculation, valid between [-0.5, 0.5]
        #        7             6         5             4          3             2
        # -1.28 x + 1.905e-13 x - 0.481 x - 6.128e-14 x - 0.6594 x + 3.753e-15 x - 0.7851 x + 0.5
        gain = polyval(self.p_coeffs, bw - self.x0)
        one = torch.ones(w0.shape)
        # generate filter coefficents
        B = torch.cat([gain, -2.0 * gain * torch.cos(w0), gain], -1)
        A = torch.cat([one, -2.0 * gain * torch.cos(w0), 2.0 * gain - 1.0], -1)
        return B, A


# iir peak filter design layer (with taylor approximation)
class IIRPeakTaylor(torch.nn.Module):
    def __init__(self, order=5, scale=0.5):
        super(IIRPeakTaylor, self).__init__()
        self.x0 = 0.5
        self.p_coeffs = self._get_polynomial(order, scale)

    def _get_polynomial(self, order, scale):
        # this is the function for K, given a normalized (0,1) bandwidth
        def f(x):
            return (np.tan(x / 2) - 1) / (np.tan(x / 2) + 1)
        # calculate taylor series coefficients
        p = approximate_taylor_polynomial(f, x=self.x0, degree=order, scale=scale)
        return p.coeffs.astype('float32')

    def forward(self, w0, bw, g):
        w0 = w0 * np.pi
        # calculate intermediate terms
        h0 = torch.pow(10.0, (g / 20.0)) - 1.0
        k = polyval(self.p_coeffs, bw - self.x0)
        l_ = -torch.cos(w0)
        one = torch.ones(w0.shape)
        # calculate filter numerator and denominator
        B = torch.cat([1 + (1 + k) * h0 / 2.0, l_ * (1 - k), -k - (1 + k) * h0 / 2.0], -1)
        A = torch.cat([one, l_ * (1 - k), -k], -1)
        return B, A


# iir filter response layer, similar to freqz
class IIRFilterResponse(torch.nn.Module):
    def __init__(self, nfft):
        super(IIRFilterResponse, self).__init__()
        self.nfft = nfft
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, b, a):
        b_fft = rfft(b, n=self.nfft, dim=1)
        a_fft = rfft(a, n=self.nfft, dim=1)
        yabs = (torch.abs(b_fft) + self.eps) / (torch.abs(a_fft) + self.eps)
        # exclude first (DC) and last (Nyquist) bins
        #y = b_fft[:, 1:-1] / a_fft[:, 1:-1]
        assert torch.all(torch.isfinite(yabs))
        # calculate absolute value of complex tuple
        #yabs = torch.abs(y)
        # reintroduce first (DC) and last (Nyquist) bins
        #ones = torch.ones(a.shape[0], 1)
        #yabs = torch.cat([ones, yabs, ones], dim=-1)
        return yabs


# iir filter response layer, similar to freqz (dB)
class IIRFilterResponseDB(torch.nn.Module):
    def __init__(self, nfft):
        super(IIRFilterResponseDB, self).__init__()
        self.resp = IIRFilterResponse(nfft)

    def forward(self, b, a):
        yabs = self.resp(b, a)
        ydb = torch.log10(yabs) * 20.0
        return ydb


# element-wise multiplication of multiple tensors
class Multiply(torch.nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return result


# modified sequential layer accepting multiple inputs
class SequentialMultiple(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
