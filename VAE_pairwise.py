import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
import matplotlib

    
    
class VAE_pairwise(nn.Module):
    
    def __init__(self, train_data, D_u = 2, n_channels = 128, n_conv = 4, n_fc = 1, d_latent = 128, 
                 eta1 = 1e3, eta2 = 2., beta_u = 1., beta_v = 1., 
                 batch_size = 32, lr = 1e-4,
                 device="cuda", filename = None):
        """
             D_u is the number of dimensions for $z^{(u)}$.
             eta1, eta2, beta_u, beta_v are the hyper-parameters for the model.
        """
        super(VAE_pairwise, self).__init__()

        self.train_data = train_data
        self.train_loader = torch.utils.data.DataLoader(self.train_data, 
                            shuffle = True, batch_size = batch_size, num_workers = 8)
            
        img_channels, self.n_pix, _ = self.train_data[0][0].shape


        if len(torch.unique(next(iter(self.train_loader))[3])) > 2:
            self.binary_similarity = False
        else:
            self.binary_similarity = True
            
        self.d_latent = d_latent
        self.n_conv = n_conv
        
        self.n_last_channels = n_channels * 2 ** (self.n_conv - 1)
        self.d_last_image = self.n_pix // 2 ** self.n_conv 
        self.d_fc = self.n_last_channels * self.d_last_image ** 2
        
        #The convolutional encoder
        self.conv = nn.ModuleList()
        self.conv_bn = nn.ModuleList()
        
        input_channel = img_channels
        output_channel = n_channels
        
        for ii in range(n_conv):
            self.conv.append(nn.Conv2d(input_channel, output_channel, 4, 2, 1))
            self.conv_bn.append(nn.BatchNorm2d(output_channel))
            input_channel = 2 ** (ii) * n_channels
            output_channel = 2 ** (ii+1) * n_channels
        
        
        #The convolutional decoder
        self.deconv = nn.ModuleList()
        self.deconv_bn = nn.ModuleList()
        
        for ii in range(n_conv):
            iii = n_conv - ii - 1
            input_channel = 2 ** (iii ) * n_channels
            output_channel = 2 ** (iii - 1) * n_channels
            
            if iii == 0:
                output_channel = img_channels

            self.deconv.append(nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1))
            self.deconv_bn.append(nn.BatchNorm2d(input_channel))
            
        # Fully connected layers for both the encoder and the ecoder.
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ii in range(n_fc):
            if ii == 0:
                self.encoders.append(nn.Linear(self.d_fc, self.d_latent))
            else:
                self.encoders.append(nn.Linear(self.d_latent, self.d_latent))
            
        self.mu_layer = nn.Linear(self.d_latent, self.d_latent)
        self.var_layer = nn.Linear(self.d_latent, self.d_latent)
        
        for ii in range(n_fc + 1):
            if ii < n_fc:
                self.decoders.append(nn.Linear(self.d_latent, self.d_latent))
            else:
                self.decoders.append(nn.Linear(self.d_latent, self.d_fc))
        
        self.D_u = D_u
        
        
        self.beta_u = beta_u
        self.beta_v = beta_v

       
        self.eta1 = eta1
        self.eta2 = eta2
           
        
        self.device = device
        
        
                                                        
        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 0.)
    
        self.n_step = 0
        self.filename = filename
        
    
    def encode(self, X):
        """
        Encode the image X.
        """
        res = X
        for ii in range(self.n_conv):
            res = F.leaky_relu_(self.conv_bn[ii](self.conv[ii](res)))
        res = res.view(-1, self.d_fc)
        for M in self.encoders:
            res = F.leaky_relu_(M(res))
        return self.mu_layer(res), self.var_layer(res)
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization.
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)
    
    def decode(self, Z):
        """
        Decode the latent representation Z.
        """
        res = Z
        for M in self.decoders[:]:
            res = F.leaky_relu_(M(res))
        
        res = res.view(-1, self.n_last_channels, self.d_last_image, self.d_last_image)
        
        for ii in range(self.n_conv - 1):
            res = F.leaky_relu_(self.deconv[ii]( self.deconv_bn[ii](res) ))

        res = torch.sigmoid( self.deconv[-1]( self.deconv_bn[-1](res) ) )
        
        return res

    def forward(self, x):
        mu, log_var = self.encode(x)
        Z = self.reparameterize(mu, log_var)
        return mu, log_var, self.decode(Z), Z
    
    
    def recon_loss(self, recon_x, x):
        """
            Reconstruction Loss (MSE).
        """
        criterion = nn.MSELoss(reduction = 'sum')
        return criterion(recon_x, x)
    
    def KLD(self, mu, log_var):
        """
            The KL divergence term.
        """
        tmp_KLD = .5 * (-1 - log_var + log_var.exp() + mu.pow(2)  )
        KLD_1 = tmp_KLD[:, :self.D_u].sum()
        KLD_2 = tmp_KLD[:, self.D_u:].sum()
        
        return self.beta_u * KLD_1 + self.beta_v * KLD_2
        #return .5 * (-1 - log_var + log_var.exp() + mu.pow(2)  ).sum()

    
    def Similarity_loss(self, Z_i, Z_j, y_ij):
        """
            The loss due to similarities.
        """
        diff = ( Z_i[:, :self.D_u] - Z_j[:, :self.D_u] ) ** 2
        t = self.eta1 * ( diff.sum(1) - self.eta2 )
        
        if self.binary_similarity:
            return - ( y_ij * F.logsigmoid( -t ) + (1 - y_ij) * F.logsigmoid( t ) ).sum()
        else:
            log_g = F.logsigmoid( t ) 
            log_g_I = F.logsigmoid( -t )
            return - ( 
                         y_ij * log_g_I + (1 - y_ij) * log_g
                         - torch.log(
                             ( 1 - 2 * torch.sigmoid( t ) ) / 
                             (log_g_I - log_g)
                         )
                     ).sum() 
        

        
    def total_loss(self, X, X_i, X_j, y_ij, alpha = 1.):
        
        N = X.shape[0]

        X_cat = torch.cat([X, X_i, X_j], 0)
        
        mu, log_var, X_recon, Z = self.forward(X_cat)
        
        Z_i = Z[N : 2 * N]
        Z_j = Z[2 * N :]
        
        C_loss = self.Similarity_loss(Z_i, Z_j, y_ij)
        loss = alpha * self.recon_loss(X_recon[:N], X) + self.KLD(mu[:N], log_var[:N]) + C_loss  
        
        return loss
    
    def train_model(self, epochs = 20, warmup_step = 0):
        
        self.train()
        
        for iii in range(epochs):
            for data_batch in self.train_loader:
                
                X = data_batch[0].to(self.device)
                X_i = data_batch[1].to(self.device)
                X_j = data_batch[2].to(self.device)
                y_ij = data_batch[3].float().to(self.device)
                
                
                batch_size = X.shape[0]
                
                self.optimizer.zero_grad()
                
                if self.n_step < warmup_step:
                    loss = self.total_loss(X, X_i, X_j, y_ij, 
                           alpha = F.relu(torch.tensor( 2 * self.n_step / warmup_step - 1 )
                                         ).to(self.device)
                                          )
                                    
                else:
                    loss = self.total_loss(X, X_i, X_j, y_ij, alpha = 1.)
                loss.backward()

                self.optimizer.step()
                
                

                if self.n_step % 500 == 0:
                    print ( "epoch:{}/{} ,step:{}, Loss: {}".format(
                                iii + 1, epochs, self.n_step, loss.item() / batch_size
                            )
                          )
                    if not self.filename is None:
                        torch.save(self.state_dict(), self.filename)
                    
                self.n_step += 1
                    

        print("Finished training.")
        if not self.filename is None:
            torch.save(self.state_dict(), self.filename)
                

    def plot_embedding(self, test_data, d1 = 0, d2 = 1):
        """
        Plot the embeddings.
        """
        z_test = None

        self.eval()

        with torch.no_grad():
            for iii in range(2000):
                x_test = test_data[iii][0].view(1, *test_data[iii][0].shape).to("cuda")
                mu, log_var = self.encode( x_test )
                Z_test = self.reparameterize(mu, log_var)

                if z_test is None:
                    z_test = Z_test[:, :].to("cpu").data.numpy()
                    y_test = test_data.y[ [iii % test_data.n_data] ]
                else:
                    z_test = np.concatenate(
                                  ( z_test, Z_test[:, :].to("cpu").data.numpy() )
                              )
                    y_test = np.concatenate(
                                  (y_test, test_data.y[ [iii % test_data.n_data] ])
                              )

        cmap=plt.get_cmap("plasma")

        z = z_test
        y = y_test

        y_max = y.max()
        y_min = y.min()
        a = np.array([[y_min,y_max]])

        font = {'size'   : 30}
        matplotlib.rc('font', **font)

        plt.figure(figsize=[8, 8])

        img = plt.imshow(a, cmap="plasma")
        plt.gca().set_visible(False)
        cax = plt.axes([0.15, 0.2, 0.75, 0.6])
        plt.colorbar( ticks=range(10) )



        for iii in range(len(z)):
            plt.plot(z[iii, d1], z[iii, d2], ".", MarkerSize = 10, 
                     Color = cmap( (y[iii] - y_min) / (y_max - y_min) ) )



    def plot_generated(self, test_data, n_img = 10, z_min = -4, z_max = 4, d1 = 0, d2 = 1):
        """
        Plot the generated images.
        """

        self.eval()


        M = transforms.ToPILImage()

        x_shape = test_data[0][0].shape
        x_test = test_data[0][0].to(self.device).view(-1, *x_shape)


        mu, log_var, recon_x, Z_test = self(x_test)

        Z = mu.data

        z_values = np.zeros([self.d_latent, n_img])


        z_values[: , :] = np.linspace(z_min, z_max, n_img)

        font = {'size'   : 30}
        matplotlib.rc('font', **font)
        
        fig = plt.figure(figsize=[8, 8])




        plt.subplots_adjust(left = .16, bottom = .15)

        ax1 = fig.add_axes([.15, .15, .76, .75])
        plt.axis('off')

        M = transforms.ToPILImage()

        idx = 0

        for ii in range(n_img):
            for jj in range(n_img):
                a = fig.add_subplot(n_img, n_img, idx + 1)
                Z[0, d2] = z_values[d1][-(ii + 1)]
                Z[0, d1] = z_values[d1][jj]
                recon = self.decode(Z).to("cpu")[0]
                #Img = M(recon[:, 20:45, 20:45])
                Img = M(recon[:])
                IMG = plt.imshow(Img, interpolation=None, cmap = "gray")
                ax = fig.gca()
                ax.set_axis_off()
                idx += 1    
        
        



