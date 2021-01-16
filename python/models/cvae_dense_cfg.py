import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch import nn
from .utils import figure_to_tensor, get_freqresp_plot


class CVAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, num_labels=1):
        super().__init__()
        assert num_labels > 0
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.latent_size = latent_size
        self.enc = Encoder(encoder_layer_sizes, latent_size, num_labels)
        self.dec = Decoder(decoder_layer_sizes, latent_size, num_labels)

    def forward(self, x, c):
        means, log_var = self.enc(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.dec(z, c)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_labels):
        super().__init__()
        layer_sizes[0] += num_labels
        self.num_labels = num_labels
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c):
        #c = idx2onehot(c, n=self.num_labels)
        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_labels):
        super().__init__()
        input_size = latent_size + num_labels
        self.num_labels = num_labels
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name=f'A{i}', module=nn.ReLU())
            #else:
            #    self.MLP.add_module(name='sigmoid', module=nn.Sigmoid())

    def forward(self, z, c):
        #c = idx2onehot(c, n=self.num_labels)
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


# dense conditional variational autoencoder
class CVAECfg(pl.LightningModule):
    model_name = 'CVAE_dense'

    def __init__(self, nfft, cfg):
        super().__init__()
        self.size_input = nfft // 2 + 1
        self.save_hyperparameters()
        self.grad_freq = 50
        self.fig_freq = 50
        self.c_labels = cfg['labels']
        encoder_layer_sizes = [self.size_input] + cfg['encoder_layer_sizes']
        latent_size = cfg['latent_size']
        decoder_layer_sizes = cfg['decoder_layer_sizes'] + [self.size_input]
        num_labels = len(self.c_labels)
        self.cvae = CVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, num_labels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nfft', type=int, default=256)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/cvae_dense/default.json')
        return parser

    def loss_function(self, resp_true, resp_pred, means, log_var, z):
        mse = torch.nn.functional.mse_loss(resp_pred, resp_true, reduction='sum') / resp_true.size(0)
        kld = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp()) / resp_true.size(0)
        loss = mse + kld
        return mse, kld, loss

    def forward(self, resp_true, c):
        return self.cvae(resp_true, c)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=20, cooldown=25),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        _, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        self.log('train_recon_loss', mse)
        self.log('train_kl', kld)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        self.log('val_recon_loss', mse)
        self.log('val_kl', kld)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        results, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        resp_pred, means, log_var, z = results
        resp_true, labels = batch
        # log metrics
        # TODO add metrics: SD
        logs = {
            'test_recon_loss': mse,
            'test_kl': kld,
            'test_loss': loss
        }
        self.log_dict(logs)
        # log reconstructions
        resp_true, resp_pred = resp_true.cpu(), resp_pred.cpu()
        labels = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
        labels = pd.DataFrame(labels)
        fig = self.get_freqresp_figure(resp_true, resp_pred, labels, n_cols=8, wh_ratio=2, width=20)
        img = figure_to_tensor(fig)
        self.logger.experiment.add_image(f'test/resp_freq_{batch_idx:04}', img, self.current_epoch)

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            # run prediction
            resp_true, c = self.example_input_array
            resp_true, c = resp_true.to(self.device), c.to(self.device)
            self.eval()
            with torch.no_grad():
                resp_pred, means, log_var, z = self.forward(resp_true, c)
            self.train()
            resp_true, resp_pred = resp_true.cpu(), resp_pred.cpu()
            # generate figure
            fig = self.get_freqresp_figure(resp_true, resp_pred, self.example_input_labels, title=True, max_rows=6)
            img = figure_to_tensor(fig)
            self.logger.experiment.add_image('Valid/resp_freq', img, self.current_epoch)

    def _shared_eval(self, batch, batch_idx):
        resp_true, labels = batch
        c = torch.stack([labels[lbl] for lbl in self.c_labels], dim=-1).float()
        results = self.forward(resp_true, c)
        losses = self.loss_function(resp_true, *results)
        return results, losses

    def get_freqresp_figure(self, resp_true, resp_pred, labels, n_cols=4, max_rows=None, width=12.0, wh_ratio=2.5, title=False):
        # setup plot
        n_rows = max(resp_true.shape[0] // n_cols, 1)
        if max_rows:
            n_rows = min(n_rows, max_rows)
        shape = (n_rows, n_cols)
        fig, axs = plt.subplots(*shape, figsize=(width, width * shape[0] / shape[1] / wh_ratio))
        for i, ax in enumerate(axs.flatten()):
            if i >= resp_true.shape[0]:
                break
            get_freqresp_plot(resp_true[i], resp_pred[i], labels.iloc[i], ax, convert_db=False)
        if title:
            fig.suptitle('Frequency responses')
            fig.tight_layout(rect=[0, 0, 1, 0.99])
        else:
            fig.tight_layout()
        return fig


# TEST CODE FOR MODEL
if __name__ == '__main__':
    import numpy as np

    nfft = 256
    cfg = {
        'encoder_layer_sizes': [64, 32],
        'decoder_layer_sizes': [32, 64],
        'latent_size': 16,
        'labels': ['el']
    }
    m = CVAECfg(nfft, cfg)

    print(m.cvae)
    print()
    print(m.cvae.enc)
    print()
    print(m.cvae.dec)

    resp_true = torch.Tensor(np.random.randn(8, nfft // 2 + 1))
    resp_pred, means, log_var, z = m.forward(resp_true)
    print(z)
    print()
    print(resp_pred.shape)
