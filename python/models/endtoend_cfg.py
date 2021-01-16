from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from .encoders import EncoderCfg
from .decoders import DecoderCfg
from .utils import get_freqresp_figure, figure_to_tensor


# dense model accounting for an arbitrary number of spectral features
class EndToEndCfg(pl.LightningModule):
    model_name = 'EtE_dense'

    def __init__(self, nfft, cfg, log_on_batch=False):
        super().__init__()
        self.save_hyperparameters()
        self.enc = EncoderCfg(nfft, cfg)
        self.dec = DecoderCfg(nfft, cfg)
        self.loss_fn = torch.nn.MSELoss()
        self.example_input_array = torch.zeros(1, nfft // 2 + 1)
        self.example_input_labels = None
        self.grad_freq = 100
        self.fig_freq = 50
        self.log_on_batch = log_on_batch

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nfft', type=int, default=256)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/ete_dense/default.json')
        parser.add_argument('--log_on_batch', action='store_true')
        return parser

    def forward(self, resp_true):
        z = self.enc(resp_true)
        resp_pred = self.dec(*z)
        return z, resp_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=20, cooldown=25),
            'monitor': 'val_early_stop_on'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        if self.log_on_batch:
            result.log('Train/loss', loss, on_step=True, on_epoch=False)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        if self.log_on_batch:
            result.log('Valid/loss', loss, on_step=True, on_epoch=False)
        return result

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result

    def training_epoch_end(self, outputs):
        # log scalars
        if not self.log_on_batch:
            avg_loss = outputs['minimize'].mean()
            self.logger.experiment.add_scalar('Train/loss', avg_loss, self.current_epoch)
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            fig = get_freqresp_figure(self, self.example_input_array, self.example_input_labels)
            img = figure_to_tensor(fig)
            self.logger.experiment.add_image('Valid/freq_resp', img, self.current_epoch)
        return outputs

    def validation_epoch_end(self, outputs):
        # log scalars
        if not self.log_on_batch:
            avg_loss = outputs['early_stop_on'].mean()
            self.logger.experiment.add_scalar('Valid/loss', avg_loss, self.current_epoch)
        return outputs

    def _shared_eval(self, batch, batch_idx):
        resp_true, _ = batch
        *_, resp_pred = self.forward(resp_true)
        loss = self.loss_fn(resp_true, resp_pred)
        return loss


# TEST CODE FOR MODEL
if __name__ == '__main__':
    import numpy as np

    nfft = 256
    cfg = {
        'spectral_features': ['notch', 'peak', 'notch'],
        'frontend': {
            'sizes': [64, 64],
            'activation_type': 'relu',
            'dropout_rate': 0.4,
        },
        'features_block': {
            'sizes': [32, 16],
            'activation_type': 'relu',
            'dropout_rate': 0.4,
        },
        'parameters_block': {
            'sizes': [8],
            'activation_type': 'none',
            'dropout_rate': 0.4,
        },
    }
    ae = EndToEndCfg(nfft, cfg)

    print(ae.enc)
    print()
    print(ae.dec)

    resp_true = torch.Tensor(np.random.randn(8, nfft // 2 + 1))
    *z, resp_pred = ae.forward(resp_true)
    print(z)
    print()
    print(resp_pred.shape)
