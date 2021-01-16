from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from .encoders import Encoder1N
from .decoders import Decoder1N
from .utils import get_freqresp_figure, figure_to_tensor


# simple model accounting for a single spectral feature
class EndToEndSimple(pl.LightningModule):
    model_name = 'EtE_simple'

    def __init__(self, nfft):
        super().__init__()
        self.save_hyperparameters()
        self.enc = Encoder1N(nfft)
        self.dec = Decoder1N(nfft)
        self.loss_fn = torch.nn.MSELoss()
        self.example_input_array = torch.zeros(1, nfft // 2 + 1)
        self.grad_freq = 3
        self.fig_freq = 3

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nfft', type=int, default=256)
        return parser

    def forward(self, resp_true):
        w0, bw = self.enc(resp_true)
        resp_pred = self.dec(w0, bw)
        return w0, bw, resp_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        result = pl.TrainResult(minimize=loss)
        #result.log('loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        #result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result

    def training_epoch_end(self, outputs):
        # log scalars
        avg_loss = outputs['minimize'].mean()
        self.logger.experiment.add_scalar('Train/loss', avg_loss, self.current_epoch)
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        return outputs

    def validation_epoch_end(self, outputs):
        # log scalars
        avg_loss = outputs['early_stop_on'].mean()
        self.logger.experiment.add_scalar('Valid/loss', avg_loss, self.current_epoch)
        # log figures
        if self.current_epoch % self.fig_freq == 0:
            fig = get_freqresp_figure(self, self.example_input_array)
            img = figure_to_tensor(fig)
            self.logger.experiment.add_image('Valid/freq_resp', img, self.current_epoch)
        return outputs

    def _shared_eval(self, batch):
        resp_true, _ = batch
        w0, bw = self.enc(resp_true)
        resp_pred = self.dec(w0, bw)
        loss = self.loss_fn(resp_true, resp_pred)
        return loss
