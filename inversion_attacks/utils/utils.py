import torch


def _initialize_x(x_shape, batch_size, device):
    """Inits the fake images

    Args:
        batch_size: the batch size

    Returns:
        randomly generated torch.Tensor whose shape is (batch_size, ) + (self.x_shape)
    """
    fake_x = torch.randn((batch_size,) + (x_shape), requires_grad=True, device=device)
    return fake_x


def _initialize_label(y_shape, batch_size, device):
    """Inits the fake labels

    Args:
        batch_size: the batch size

    Returns:
        randomly initialized or estimated labels
    """
    fake_label = torch.randn((batch_size, y_shape), requires_grad=True, device=device)
    fake_label = fake_label.to(device)
    return fake_label


def _estimate_label(received_gradients, batch_size, pos_of_final_fc_layer, device):
    """Estimates the secret labels from the received gradients

    this function is based on the following papers:
    batch_size == 1: https://arxiv.org/abs/2001.02610
    batch_size > 1: https://arxiv.org/abs/2104.07586

    Args:
        received_gradients: gradients received from the client
        batch_size: batch size used to culculate the received_gradients

    Returns:
        estimated labels
    """
    if batch_size == 1:
        fake_label = torch.argmin(
            torch.sum(received_gradients[pos_of_final_fc_layer], dim=1)
        )
    else:
        fake_label = torch.argsort(
            torch.min(received_gradients[pos_of_final_fc_layer], dim=-1)[0]
        )[:batch_size]
    fake_label = fake_label.reshape(batch_size)
    fake_label = fake_label.to(device)
    return fake_label


def _setup_attack(
    x_shape,
    y_shape,
    optimizer_class,
    optimize_label,
    pos_of_final_fc_layer,
    device,
    received_gradients,
    batch_size,
    init_x=None,
    labels=None,
    **kwargs
):
    """Initializes the image and label, and set the optimizer

    Args:
        received_gradients: a list of gradients received from the client
        batch_size: the batch size

    Returns:
        initial images, labels, and the optimizer instance
    """
    fake_x = _initialize_x(x_shape, batch_size, device) if init_x is None else init_x
    # if init_x is None , it is duplicated
    fake_x.requires_grad = True

    if labels is None:
        fake_label = (
            _initialize_label(y_shape, batch_size, device)
            if optimize_label
            else _estimate_label(
                received_gradients,
                batch_size,
                pos_of_final_fc_layer,
                device,
            )
        )
    else:
        fake_label = labels

    optimizer = (
        optimizer_class([fake_x, fake_label], **kwargs)
        if optimize_label
        else optimizer_class(
            [
                fake_x,
            ],
            **kwargs,
        )
    )

    return fake_x, fake_label, optimizer


def _generate_fake_gradients(
    target_model, lossfunc, optimize_label, fake_x, fake_label, image_prior_model=None
):
    fake_pred = target_model(fake_x)
    if optimize_label:
        loss = lossfunc(fake_pred, fake_label.softmax(dim=-1))
    else:
        loss = lossfunc(fake_pred, fake_label)
    fake_gradients = torch.autograd.grad(
        loss,
        target_model.parameters(),
        create_graph=True,
        #allow_unused=True,
    )
    if image_prior_model:
        image_prior_model(fake_x,fake_x)
    return fake_pred, fake_gradients



#####



import random
import numpy as np
from torch.utils.data.dataset import Dataset


def try_gpu(e):
    """Send given tensor to gpu if it is available

    Args:
        e: (torch.Tensor)

    Returns:
        e: (torch.Tensor)
    """
    if torch.cuda.is_available():
        return e.cuda()
    return e


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RoundDecimal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_digits):
        ctx.save_for_backward(input)
        ctx.n_digits = n_digits
        return torch.round(input * 10**n_digits) / (10**n_digits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return torch.round(grad_input * 10**ctx.n_digits) / (10**ctx.n_digits), None


torch_round_x_decimal = RoundDecimal.apply


class NumpyDataset(Dataset):
    """This class allows you to convert numpy.array to torch.Dataset

    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):

    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y=None, transform=None, return_idx=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        x = self.x[index]
        if self.y is not None:
            y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if not self.return_idx:
            if self.y is not None:
                return x, y
            else:
                return x
        else:
            if self.y is not None:
                return index, x, y
            else:
                return index, x

    def __len__(self):
        """get the number of rows of self.x"""
        return len(self.x)



class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    validate : int
    warmup: bool
    dryrun : bool
    dropout : float
    augmentations : bool

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Defaulted parameters. Apply overwrites from args."""
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr
        if dryrun:
            self.dryrun = dryrun
        self.validate = 10

class ConservativeStrategy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 120 if epochs is None else epochs
        self.batch_size = 8 #128
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.warmup = False
        self.weight_decay : float = 5e-4
        self.dropout = 0.0
        self.augmentations = False #True
        self.dryrun = False
        super().__init__(lr=None, epochs=None, dryrun=False)


#### mine
import matplotlib.pyplot as plt
def plot(data,ds,dm,font_size=25,fig_size=(7,8)):
    img =  data.clone().detach().to(data.device)
    plt.axis('off')
    plt.rcParams.update({'font.size': font_size})
    img.mul_(ds).add_(dm).clamp_(0, 1)
    if img.shape[0] == 1:
        return plt.imshow(img[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, img.shape[0], figsize=(12, img.shape[0]*12))
        for i, im in enumerate(img):
            axes[i].imshow(im.permute(1, 2, 0).cpu())


def multi_plot(data,ds,dm,labels=None,classes=None):

    img =  data.clone().detach().to(data.device)

    grid_shape = 2 #int(torch.as_tensor(data.shape[0]).sqrt().ceil())
    s = 10 if data.shape[3] > 150 else 6
    fig, axes = plt.subplots(grid_shape, int(data.shape[0]/grid_shape), figsize=(s, int(s*2/3)))
    plt.rcParams.update({'font.size': 9})

    img.mul_(ds).add_(dm).clamp_(0, 1)
    img = img.to(dtype=torch.float32)

    label_classes = []
    for i, (im, axis) in enumerate(zip(img, axes.flatten())):
        axis.imshow(im.permute(1, 2, 0).cpu())
        if labels is not None :
            label_classes.append(classes[labels[i]])
            axis.set_title(str(classes[labels[i]][0]))
        axis.axis("off")
    # if labels is not None:
    #     print(label_classes)