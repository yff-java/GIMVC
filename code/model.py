from torch import nn


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, z_dim),
        )

    def forward(self, x):
        z = self.linears(x)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim):
        super(Decoder, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, x_dim),
        )

    def forward(self, z):
        x = self.linears(z)
        return x


class DegradationModel(nn.Module):
    def __init__(self, h_dim, x_dim_list):
        super(DegradationModel, self).__init__()
        self.decoder_list = []
        for x_dim in x_dim_list:
            self.decoder_list.append(Decoder(h_dim, x_dim))
        self.decoder_list = nn.ModuleList(self.decoder_list)

    def forward(self, h):
        x_re_list = []
        for decoder in self.decoder_list:
            x_re_list.append(decoder(h))
        return x_re_list


class AutoEncoderMV(nn.Module):
    def __init__(self, x_dim_list, z_dim):
        super(AutoEncoderMV, self).__init__()
        self.encoder_list = [Encoder(x_dim, z_dim) for x_dim in x_dim_list]
        self.encoder_list = nn.ModuleList(self.encoder_list)
        self.decoder_list = [Decoder(z_dim, x_dim) for x_dim in x_dim_list]
        self.decoder_list = nn.ModuleList(self.decoder_list)

    def forward(self, x):
        z_list = []
        x_re_list = []
        for x_i, encoder in zip(x, self.encoder_list):
            z_list.append(encoder(x_i))
        for z_i, decoder in zip(z_list, self.decoder_list):
            x_re_list.append(decoder(z_i))
        return z_list, x_re_list
