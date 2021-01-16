import io
import librosa as lr
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

# convert matplotlib figure to tensor
def figure_to_tensor(fig):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    # Closing the figure
    plt.close(fig)
    buf.seek(0)
    # Convert buffer to tensor
    img = Image.open(buf)
    return ToTensor()(img)

# generate figure with multiple frequency responses
def get_freqresp_figure(model, resps_true, resps_labels, shape=(4, 4)):
    params_names = [f'{k}_{p}' for k in model.enc.features for p in model.enc.features[k]]
    # run prediction
    with torch.no_grad():
        *z, resps_pred = model.forward(resps_true)
    # setup plot
    fig, axs = plt.subplots(*shape, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        values = [p.detach().numpy()[i, 0] for k in z for p in k]
        params = zip(params_names, values)
        get_freqresp_plot(resps_true[i], resps_pred[i], resps_labels.iloc[i], ax, params, convert_db=False)
    fig.suptitle('Frequency responses')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

# generate and format a single plot
def get_freqresp_plot(resp_true, resp_pred, lbl, ax, params=None, convert_db=True):
    # detach tensors from computational graph
    x_true = resp_true.detach().numpy() if resp_true.requires_grad else resp_true.numpy()
    x_pred = resp_pred.detach().numpy() if resp_pred.requires_grad else resp_pred.numpy()
    title = '{subj}-{ear} ({az}, {el})'.format(**lbl)
    if convert_db:
        ax.plot(lr.amplitude_to_db(x_true))
        ax.plot(lr.amplitude_to_db(x_pred))
        ax.set_ylim([-90, 10])
        text_y = 0
    else:
        ax.plot(x_true)
        ax.plot(x_pred)
        #ax.set_ylim([0.5, 3.5])
        text_y = 1
    ax.set_xlim([0, len(x_true)])
    ax.set_title(title)
    # show spectral params z
    if params:
        label_str = '\n'.join([f'{n}: {p:.3f}' for n, p in params])
        ax.text(0, text_y, label_str)
