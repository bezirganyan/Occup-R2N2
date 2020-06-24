import torch
import torch.nn as nn
from torch.autograd import Variable
from im2mesh.r2n2.models.decoder import Decoder


# Decoder dictionary
decoder_dict = {
    'simple': Decoder,
}


class R2N2(nn.Module):
    ''' The 3D Recurrent Reconstruction Neural Network (3D-R2N2) model.

    For details regarding the model, please see
    https://arxiv.org/abs/1604.00449

    As single-view images are used as input, we do not use the recurrent
    module.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
    '''

    def __init__(self, decoder, encoder, h_shape):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.h_shape = h_shape

    def forward(self, x):
        c = self.encode_inputs(x)
        occ_hat = self.decoder(c)
        return occ_hat

    def encode_inputs(self, x):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        #initialize the hidden state and update gate
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)

        #a list used to store intermediate update gate activations
        u_list = []

        """
        x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network.
        """
        x = x.transpose(0, 1)
        for time in range(x.size(0)):
            gru_out, update_gate = self.encoder(x[time], h, u, time)

            h = gru_out

            u = update_gate
            u_list.append(u)

        return h.view(x.size(1), -1)

    def initHidden(self, h_shape):
        h = torch.zeros(h_shape)
        if torch.cuda.is_available():
            h = h.cuda()
        return Variable(h)
