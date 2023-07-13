import torch
from torch import nn

def gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    epsilon: float, norm: str = "2",
                    loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack (PGD) on the input x.

    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is 
            the number of classes. 
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation 
        (before the projection step) will have a norm of exactly epsilon as 
        measured by the desired norm (see argument: norm). Therefore, epsilon
        implicitly fixes the step size of the PGD update.
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", 
        the perturbation (before the projection step) will have a L_1 norm of 
        exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is 
        simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.
    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]
    ##########################################################
    # YOUR CODE HERE
    loss = loss_fn(logits, y)

   # gradients = torch.autograd.grad(loss, x)[0]

    if norm == "inf":
        perturbation = epsilon * torch.sign(gradients)
    else:
        gradients_norm = torch.norm(gradients.view(gradients.shape[0], -1), p=float(norm), dim=1)
        perturbation = (epsilon / gradients_norm)[:, None, None, None] * gradients

        # Add perturbation to the input
    x_pert = x + perturbation
    ##########################################################
    return x_pert.detach()


def attack(x: torch.Tensor, y: torch.Tensor, model: nn.Module, attack_args: dict):
    """
    Run the gradient_attack function above once on x.

    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    model: nn.Module
        The model to be attacked.
    attack_args: dict 
        Additional arguments to be passed to the attack function.

    Returns
    -------
    x_pert: torch.Tensor of the same shape of x 
        Similar as x but perturbed
    y_pert:  torch.Tensor of shape [B, 1]
        Predictions for x_pert
    """

    ##########################################################
    # YOUR CODE HERE

    logits = model(x)
    x.requires_grad = False
    y.requires_grad = False

    x_pert = gradient_attack(logits, x, y, **attack_args)
    y_pert = torch.argmax(model(x_pert), dim=1)
    ##########################################################
    return x_pert, y_pert
