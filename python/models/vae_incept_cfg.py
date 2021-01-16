import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from .vae_incept import VAE


# convolutional variational autoencoder with inception layers
class InceptionVAECfg(pl.LightningModule):
    model_name = 'VAE_incept'

    def __init__(self, input_size, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.grad_freq = 50
        self.fig_freq = 10
        self.kl_coeff = cfg['kl_coeff']
        # init model
        latent_size = cfg['latent_size']
        use_inception = cfg['use_inception']
        repeat_per_block = cfg['repeat_per_block']
        self.vae = VAE(latent_size=latent_size, use_inception=use_inception, repeat_per_block=repeat_per_block)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=30, cooldown=25),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/vae_incept/18.json')
        return parser

    def forward(self, x):
        return self.vae(x)

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            # run prediction
            ear_true = self.example_input_array.to(self.device)
            self.eval()
            with torch.no_grad():
                ear_pred, means, log_var = self.forward(ear_true)
            self.train()
            ear_true = ear_true.to(self.device)
            # generate figure
            img = self.get_pred_ear_figure(ear_true, ear_pred)
            self.logger.experiment.add_image('Valid/ears', img, self.current_epoch)

    def get_pred_ear_figure(self, ear_true, ear_pred, n_cols=8):
        bs = ear_true.shape[0]
        n_rows = max(bs, 1) // n_cols
        imgs = []
        for i in range(n_rows):
            sl = slice(i * n_cols, min((i + 1) * n_cols, bs))
            img_true = torch.dstack(ear_true[sl].unbind())
            img_pred = torch.dstack(ear_pred[sl].unbind())
            img = torch.hstack((img_true, img_pred))
            imgs.append(img)
        img = torch.hstack(imgs)
        return img

    def loss_function(self, x, recon_x, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * self.kl_coeff
        return BCE, KLD, BCE + KLD

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
        ear_pred, means, log_var = results
        ear_true, labels = batch
        # log metrics
        # TODO add metrics: SD
        logs = {
            'test_recon_loss': mse,
            'test_kl': kld,
            'test_loss': loss
        }
        self.log_dict(logs)
        # log reconstructions
        ear_true, ear_pred = ear_true.cpu(), ear_pred.cpu()
        img = self.get_pred_ear_figure(ear_true, ear_pred, n_cols=8)
        self.logger.experiment.add_image(f'test/ears_{batch_idx:04}', img, self.current_epoch)

    def _shared_eval(self, batch, batch_idx):
        ear_true, labels = batch
        results = self.forward(ear_true)
        losses = self.loss_function(ear_true, *results)
        return results, losses
