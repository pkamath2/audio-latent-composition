import collections as c
import torch
import torch.fft
import numpy as np

class GMLoss(torch.nn.Module):
    def __init__(self, loss_type='autocorrelation'):
        super(GMLoss, self).__init__()

        self._loss_types = ['autocorrelation', 'l2', 'cosine']
        assert loss_type in self._loss_types

        self.loss_type = loss_type
        self.N_PARALLEL_CNNS=6
        self.N_FFT = 512 
        self.K_HOP = 128 
        self.N_FREQ= 256
        self.N_FILTERS = 512

        possible_kernels = [2,4,8,16,64,128,256,512,1024,2048]
#         possible_kernels = [8,16,64,128,256,512,1024,2048]
        self.filters = [0]*self.N_PARALLEL_CNNS
        for i in range(self.N_PARALLEL_CNNS):
            self.filters[i]=possible_kernels[i]
        
        if loss_type == 'cosine':
            self.cosine_d = torch.nn.CosineSimilarity(dim=0)

        #Initialize CNNs
        self.cnnlist=[]
        for i in range(self.N_PARALLEL_CNNS):
            cnn_layer = torch.nn.Sequential(c.OrderedDict([
                                ('conv1',torch.nn.Conv2d(self.N_FREQ, self.N_FILTERS, kernel_size=(1,self.filters[i]),bias=False)),
                                ('relu1',torch.nn.ReLU())]))
            cnn_layer.apply(lambda x: self._weights_init(x))
            for param in cnn_layer.parameters():
                param.requires_grad = False
            cnn_layer.cuda()
            self.cnnlist.append(cnn_layer)
    
    # Glorot initialization
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight)
    
    #Compute the feature matrix
    def _gm_features(self, im, layer_num):
        f = self.cnnlist[layer_num](im)
        f = f.permute(0,2,1,3)
        return f
    
    #Compute the feature correlations
    def _gm(self, inp_features):
        batch_size, c, h, w = inp_features.size()                  # batch_size X 1 X 512 X 512      
        features = inp_features.squeeze(1)                         # batch_size X 512 X 512 
        grammatrix = torch.bmm(features, features.transpose(1,2)).div(c*h*w)
        grammatrix = grammatrix.reshape(batch_size, 512, 512)
        return grammatrix

    
    def forward(self, im0, im1): #im0/im1.shape = batch_size X 1 X 256 X 256

        n,c,h,w = im0.shape
        im0 = im0.permute(0,2,1,3)
        im1 = im1.permute(0,2,1,3)

        metric_arr = []
        nume = torch.zeros((n,1)).cuda()
        deno = torch.zeros((n,1)).cuda()
        for layer_num in range(self.N_PARALLEL_CNNS):
            gm0 = self._gm_features(im0, layer_num)
            gm1 = self._gm_features(im1, layer_num)

            gm0 = self._gm(gm0)
            gm1 = self._gm(gm1)

            if self.loss_type == 'autocorrelation':
                gm0_fft = torch.fft.fftn(gm0, dim=-2)
                gm1_fft = torch.fft.fftn(gm1, dim=-2)
                
                gm0_conj_fft = torch.conj(gm0_fft)
                gm1_conj_fft = torch.conj(gm1_fft)
                
                gm0_ifft = torch.abs(torch.fft.ifft(torch.bmm(gm0_fft, gm0_conj_fft)))
                gm1_ifft = torch.abs(torch.fft.ifft(torch.bmm(gm1_fft, gm1_conj_fft)))
                gm0_ifft = torch.abs(torch.fft.ifft(torch.bmm(gm0_fft, gm0_conj_fft)))
                gm1_ifft = torch.abs(torch.fft.ifft(torch.bmm(gm1_fft, gm1_conj_fft)))
                
                n,h,w = gm0_ifft.shape
                x = torch.square(gm0_ifft - gm1_ifft)
                x_mean = x.view(n, -1).mean(1, keepdim=True)
                nume += x_mean

                y = torch.square(gm1_ifft)
                y_mean = y.view(n, -1).mean(1, keepdim=True)
                deno += y_mean
            
            if self.loss_type == 'l2':
                metric_arr.append(torch.dist(gm0, gm1, 2))
                    
            if self.loss_type == 'cosine':
                metric_arr.append(self.cosine_d(gm0, gm1))

        if self.loss_type == 'autocorrelation':
            return torch.mean(nume/deno)
        else:
            return torch.mean(torch.stack(metric_arr))

