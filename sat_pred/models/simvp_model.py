"""Adapted from https://github.com/A4Bio/SimVP
"""

import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        transpose=False, 
        act_norm=False
    ):
        super(BasicConv2d, self).__init__()
        
        if transpose:
            conv_layer = nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                output_padding=stride //2 
            )
        else:
            conv_layer = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            )
            
        layers = [conv_layer]
            
        if act_norm:
            layers.append(nn.GroupNorm(2, out_channels))
            layers.append(nn.LeakyReLU(0.2))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        
        if stride == 1:
            transpose = False
            
        self.model = BasicConv2d(
            C_in, 
            C_out, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            transpose=transpose, 
            act_norm=act_norm
        )

    def forward(self, x):
        return self.model(x)


class GroupConv2d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        groups, 
        act_norm=False
    ):
        super(GroupConv2d, self).__init__()        
        
        if in_channels % groups != 0:
            groups = 1
            
        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                groups=groups
            )
        ]
        
        if act_norm:
            layers.append(nn.GroupNorm(groups, out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for kernel_size in incep_ker:
            layers.append(
                GroupConv2d(
                    C_hid, 
                    C_out, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size//2, 
                    groups=groups, 
                    act_norm=True
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


# I think this might have a problem for odd values of N when reverse=True
def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: 
        return list(reversed(strides[:N]))
    else: 
        return strides[:N]

    
def stride_generator_new(N, reverse=False):
    if reverse:
        strides = [2, 1]
    else:
        strides = [1, 2]
    
    return (strides*((N+1)//2))[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        
        layers = [ConvSC(C_in, C_hid, stride=strides[0])]
        layers.extend([ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]])
        
        self.encoder_layers = nn.ModuleList(layers)
    
    def forward(self, x):# B*4, 3, 128, 128
        enc1 = self.encoder_layers[0](x)
        latent = enc1
        for layer in self.encoder_layers[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)        
        
        layers = [ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]]
        layers.append(ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True))
        
        self.decoder_layers = nn.ModuleList(layers)
        
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.decoder_layers)-1):
            hid = self.decoder_layers[i](hid)
        hid = hid[..., :enc1.shape[-2], :enc1.shape[-1]]
        Y = self.decoder_layers[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

    
class Mid_Xnet(nn.Module):
    
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y
    

class Mid_Xnet_Temporal(nn.Module):
    
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet_Temporal, self).__init__()

        self.time_embedding = nn.Sequential(
            nn.Linear(4, channel_hid),
            nn.ReLU(),
            nn.Linear(channel_hid, channel_hid),
        )

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, time_features):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # time stuff
        # (reminding myself that T = num history steps)
        # initial size: [B, T, 4]
        t = self.time_embedding(time_features)  # [B, T, hidden]
        t = t.unsqueeze(-1).unsqueeze(-1)       # [B, T, hidden, 1, 1]
        t = t.expand(-1, -1, -1, H, W)          # [B, T, hidden, H, W]
        t = t.reshape(B, -1, H, W)              # [B, T*hidden, H, W]

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z + t)  # add time embedding to result
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y



class SimVP(nn.Module):
    def __init__(
        self, 
        num_channels, 
        history_len, 
        forecast_len, 
        spatial_size=(279, 386), 
        hid_S=16, 
        hid_T=256, 
        N_S=4,
        N_T=8, 
        incep_ker=[3,5,7,11], 
        groups=8,
    ):
        super(SimVP, self).__init__()
                
        self.enc = Encoder(num_channels, hid_S, N_S)
        self.hid = Mid_Xnet(history_len*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, num_channels, N_S)
        self.spatial_size = spatial_size


    def forward(self, x_raw):
        
        # Pad out to a multiple of downsample factor
        #pad_top = pad_left = 0
        #downsample_factor = (N_S // 2)*2
        #pad_bottom = downsample_factor - (self.spatial_size[0] % downsample_factor)
        #pad_right = downsample_factor - (self.spatial_size[1] % downsample_factor)
        #x_raw = F.pad(x_raw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # (batch, channel, time, height, width) -> (batch, time, channel, height, width)
        x_raw = x_raw.permute(0,2,1,3,4)
        
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        
        Y = Y.permute(0,2,1,3,4)

        # Remove padding
        # Y = Y[..., :self.spatial_size[0]-pad_bottom, :self.spatial_size[1]-pad_right]
        return Y
    

class SimVPTemporal(nn.Module):
    def __init__(
        self, 
        num_channels, 
        history_len, 
        forecast_len, 
        spatial_size=(279, 386), 
        hid_S=16, 
        hid_T=256, 
        N_S=4,
        N_T=8, 
        incep_ker=[3,5,7,11], 
        groups=8,
    ):
        super(SimVPTemporal, self).__init__()                
        self.enc = Encoder(num_channels, hid_S, N_S)
        self.hid = Mid_Xnet_Temporal(history_len*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, num_channels, N_S)
        self.spatial_size = spatial_size


    def forward(self, features):
 
        x_raw, time_features = features
        
        # Pad out to a multiple of downsample factor
        #pad_top = pad_left = 0
        #downsample_factor = (N_S // 2)*2
        #pad_bottom = downsample_factor - (self.spatial_size[0] % downsample_factor)
        #pad_right = downsample_factor - (self.spatial_size[1] % downsample_factor)
        #x_raw = F.pad(x_raw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # (batch, channel, time, height, width) -> (batch, time, channel, height, width)
        x = x_raw.permute(0,2,1,3,4)
        
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z, time_features)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        
        Y = Y.permute(0,2,1,3,4)

        # Remove padding
        # Y = Y[..., :self.spatial_size[0]-pad_bottom, :self.spatial_size[1]-pad_right]
        return Y