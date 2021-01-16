import torch
from argparse import ArgumentParser
from pl_bolts.models.autoencoders import VAE


# convolutional variational autoencoder with residual layers
class ResNetVAECfg(VAE):
    model_name = 'VAE_resnet'

    def __init__(self, input_size, cfg):
        self.save_hyperparameters()
        self.grad_freq = 50
        self.fig_freq = 10
        input_height = input_size[0]
        # TODO implement from scratch so it only uses 1 channel
        super().__init__(
            input_height,
            enc_type=cfg['enc_type'],
            enc_out_dim=cfg['enc_out_dim'],
            latent_dim=cfg['latent_size'],
            kl_coeff=cfg['kl_coeff'],
            first_conv=cfg['first_conv'])

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
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/vae_resnet/18.json')
        return parser

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            img = self.get_pred_ear_figure(self.example_input_array, self.example_input_labels)
            self.logger.experiment.add_image('Valid/ears', img, self.current_epoch)

    def get_pred_ear_figure(self, ear_true, labels, n_images=6):
        ear_true = ear_true.to(self.device)
        # run prediction
        self.eval()
        with torch.no_grad():
            ear_pred = self.forward(ear_true)
        self.train()
        img_true = torch.dstack(ear_true[:n_images].unbind())
        img_pred = torch.dstack(ear_pred[:n_images].unbind())
        img = torch.hstack((img_true, img_pred))
        return img
