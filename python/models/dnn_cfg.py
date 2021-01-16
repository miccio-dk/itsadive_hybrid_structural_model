import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch import nn


class DNN(nn.Module):
    def __init__(self, input_size, outputs_size, hidden_layers, dropout_rate=0.2):
        super().__init__()

        layers = [input_size, *hidden_layers, outputs_size]
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            if i < len(layers) - 2:
                lb = DNN.linear_block(in_size, out_size, dropout_rate=dropout_rate)
            else:
                lb = DNN.linear_block(in_size, out_size, activation=None)
            self.seq.add_module(name=f'block_{i}', module=lb)

    @staticmethod
    def linear_block(in_size, out_size, activation=nn.LeakyReLU, dropout_rate=None):
        lb = nn.Sequential()
        lb.add_module(name='lin', module=nn.Linear(in_size, out_size))
        if activation:
            lb.add_module(name='act', module=activation())
        if dropout_rate:
            lb.add_module(name='drop', module=nn.Dropout(dropout_rate))
        return lb

    def forward(self, x):
        return self.seq(x)


# dense deep neural network
class DNNCfg(pl.LightningModule):
    model_name = 'DNN'

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.grad_freq = 50
        self.fig_freq = 10
        self.c_labels = cfg['c']
        input_size = cfg['z_ears_size'] + len(self.c_labels)
        outputs_size = cfg['z_hrtf_size']
        hidden_layers = cfg['hidden_layers']
        dropout_rate = cfg['dropout_rate']
        self.dnn = DNN(input_size, outputs_size, hidden_layers, dropout_rate=dropout_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/dnn/small.json')
        return parser

    def loss_function(self, z_hrtf_pred, z_hrtf_true):
        mse = torch.nn.functional.mse_loss(z_hrtf_pred, z_hrtf_true, reduction='mean')
        return mse

    def forward(self, x):
        return self.dnn(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=50, cooldown=25),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        _, loss = self._shared_eval(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._shared_eval(batch, batch_idx)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        z_hrtf_pred, loss = self._shared_eval(batch, batch_idx)
        z_ears, z_hrtf_true, labels = batch
        # log metrics
        # TODO add metrics: R^2
        self.log('test_loss', loss)
        # log prediction
        # TODO generate confusion matrix?
        # z_hrtf_true, z_hrtf_pred = z_hrtf_true.cpu(), z_hrtf_pred.cpu()
        # img = self.get_pred_ear_figure(ear_true, ear_pred, n_cols=8)
        # self.logger.experiment.add_image(f'test/ears_{batch_idx:04}', img, self.current_epoch)

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        # log figures
        # if self.current_epoch % self.fig_freq == 0:
        #     # run prediction
        #     z_ears = self.example_input_array.to(self.device)
        #     self.eval()
        #     with torch.no_grad():
        #         z_hrtf_pred = self.forward(z_ears)
        #     self.train()
        #     # generate figure
        #     z_hrtf_pred = z_hrtf_pred.to(self.device)
        #     img = self.get_pred_ear_figure(ear_true, ear_pred)
        #     self.logger.experiment.add_image('Valid/ears', img, self.current_epoch)

    def _shared_eval(self, batch, batch_idx):
        z_ears, z_hrtf_true, labels = batch
        # concatenate labels to z_ears
        c = torch.stack([labels[lbl] for lbl in self.c_labels], dim=-1).float()
        x = torch.cat((z_ears, c), dim=-1)
        # do forward pass and calculate loss
        z_hrtf_pred = self.forward(x)
        loss = self.loss_function(z_hrtf_pred, z_hrtf_true)
        return z_hrtf_pred, loss

    def get_confusion_matrix(self, z_hrtf_pred, z_hrtf_true):
        pass
