import numpy as np
import torch
import torch.nn as nn
from .enc_dec import Filterbank


class Param_GCFB(Filterbank):
    """ 
    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        sample_rate (int, optional): The sample rate (used for initialization).
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.
        
    p: order of gammatone (default = 4)
    fc: center frequency
    b: bandwidth
    phi: phase shift 
    c: chirp term (default = -1)
    """
    
    def __init__(self, n_filters=128, kernel_size=16, sample_rate=16000, stride=None, min_low_hz=50, min_band_hz=50, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.n_feats_out = n_filters
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        
        _t_ = (torch.arange(1.0, self.kernel_size + 1).view(1, -1) / self.sample_rate)
                
        # filter order
        _p_ = torch.tensor(4.0)  # order
        
        # intiliazation 
        low_hz = 50.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        # linear spaced in ERB scale
        erb_f = np.linspace(
            self.freq_hz_2_erb_scale(low_hz), self.freq_hz_2_erb_scale(high_hz), self.n_filters, dtype="float32"
        )  
        hz = self.erb_scale_2_freq_hz(erb_f) 
                        
        erb = 24.7 + 0.108 * hz  # equivalent rectangular bandwidth
        divisor = (np.pi * np.math.factorial(2 * _p_ - 2) * np.power(2, float(-(2 * _p_ - 2)))) / np.square(
                np.math.factorial(_p_ - 1))
        _b_ = erb / divisor  # bandwidth parameter
        
        _phi_ = np.zeros(self.n_filters,dtype="float32")   
        _c_ = -1*np.ones(self.n_filters,dtype="float32")
        
        self.register_buffer("_t_", _t_)
        
        # filters parameters         
        self.p = nn.Parameter(_p_.view(-1, 1))
        self.fc = nn.Parameter(torch.from_numpy(hz).view(-1, 1))       
        self.b = nn.Parameter(torch.from_numpy(_b_).view(-1, 1))
        self.phi = nn.Parameter(torch.from_numpy(_phi_).view(-1, 1))
        self.c = nn.Parameter(torch.from_numpy(_c_).view(-1, 1))

    @property
    def filters(self):
        
        eps = 1e-6
        
        phi_compensation = -self.fc * (self.p - 1) / (self.b+eps)    
    
        A = (4.0*np.pi*self.b)**((2.0*self.p+1.0)/2.0)/torch.sqrt(torch.exp(torch.lgamma(2.0*self.p+1.0)))*np.sqrt(2.0) # normalization
        gtone = self._t_**(self.p-1)*torch.exp(-2*np.pi*self.b*self._t_)*torch.cos(2*np.pi*self.fc*self._t_ + self.c*torch.log(self._t_) + phi_compensation + self.phi)
        gtone = A * gtone
        normalization_value = 1.0 / torch.sqrt(torch.mean(torch.pow(gtone,2), dim=1))  # rms
        normalization_gtone = gtone * normalization_value[:, np.newaxis]        
        return normalization_gtone.view(self.n_filters, 1, self.kernel_size)

    
    @staticmethod
    def erb_scale_2_freq_hz(freq_erb):
        """ Convert frequency on ERB scale to frequency in Hertz """
        freq_hz = (np.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
        return freq_hz
    
    @staticmethod
    def freq_hz_2_erb_scale(freq_hz):
        """ Convert frequency in Hertz to frequency on ERB scale """
        freq_erb = 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
        return freq_erb 
