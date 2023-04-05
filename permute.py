import torch
import torch.nn as nn
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""
Reference:

Peña, Fidel A. Guerrero, et al. "Re-basin via implicit Sinkhorn differentiation." 
arXiv preprint arXiv:2212.12042 (2022).
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""

# Implicit Sinkhorn
class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b
            log_p -= torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        device = grad_p.device

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat(
            (
                torch.cat((torch.diag_embed(a), p), dim=-1),
                torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat(
            (grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1
        ).unsqueeze(-1)

        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (
                grad_ab[..., m:, :],
                torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32),
            ),
            dim=-2,
        )
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


# Linear Assignment Problem
def matching(alpha, **kwargs):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha, **kwargs)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)


class ReparamNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.output = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        for p1, p2 in zip(self.model.parameters(), self.output.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def set_model(self, model):
        self.model = copy.deepcopy(model)
        for p1 in self.model.parameters():
            p1.requires_grad = False

    def forward(self, P):
        for p1, p2 in zip(self.output.parameters(), self.model.parameters()):
            p1.data = p2.data.clone()

        for p1 in self.output.parameters():
            p1._grad_fn = None

        i = 0
        for (name, p1), p2 in zip(
            self.output.named_parameters(), self.model.parameters()
        ):

            if "bias" in name:
                i -= 1
                p1.copy_(P[i] @ p2)

            # bacthnorm
            elif len(p1.shape) == 1 and p1.shape[0] == P[i - 1].shape[0]:
                i -= 1
                p1.copy_((P[i] @ p2.view(p1.shape[0], -1)).view(p2.shape))

            # mlp / cnn
            elif "weight" in name:
                if i < len(P) and i == 0:
                    p1.copy_((P[i] @ p2.view(P[i].shape[0], -1)).view(p2.shape))

                if i < len(P) and i > 0:
                    p1.copy_(
                        (
                            P[i - 1].view(1, *P[i - 1].shape)
                            @ (P[i] @ p2.view(P[i].shape[0], -1)).view(
                                p2.shape[0], P[i - 1].shape[0], -1
                            )
                        ).view(p2.shape)
                    )

                if i == len(P) and i > 0:
                    p1.copy_(
                        (
                            P[i - 1].view(1, *P[i - 1].shape)
                            @ p2.view(p2.shape[0], P[i - 1].shape[0], -1)
                        ).view(p2.shape)
                    )

            i += 1
            if i > len(P):
                break

        return self.output

    def to(self, device):
        self.output.to(device)
        self.model.to(device)

        return self


class RebasinNet(torch.nn.Module):
    def __init__(
        self, model, P_sizes=None, l=1.0, tau=1.0, n_iter=20, operator="implicit"
    ):
        super().__init__()
        assert operator in [
            "implicit",
        ], "Operator must be either `implicit`"

        self.reparamnet = ReparamNet(model)

        if P_sizes is None:
            P_sizes = list()
            for name, p in model.named_parameters():
                if "weight" in name:
                    if len(p.shape) == 1:  # batchnorm
                        pass  # no permutation : bn is "part" for the previous one like biais
                    else:
                        P_sizes.append((p.shape[0], p.shape[0]))
            P_sizes = P_sizes[:-1]

        self.p = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.eye(ps[0]) + torch.randn(ps) * 0.1, requires_grad=True
                )
                for ps in P_sizes
            ]
        )

        self.l = l
        self.tau = tau
        self.n_iter = n_iter
        self.operator = operator

    def random_init(self):
        for p in self.p:
            ci = torch.randperm(p.shape[0])
            p.data = (torch.eye(p.shape[0])[ci, :]).to(p.data.device)

    def identity_init(self):
        for p in self.p:
            p.data = torch.eye(p.shape[0]).to(p.data.device)

    def forward(self, x=None):

        if self.training:
            gk = list()
            for i in range(len(self.p)):
                if self.operator == "implicit":
                    sk = Sinkhorn.apply(
                        -self.p[i] * self.l,
                        torch.ones((self.p[i].shape[0])).to(self.p[0].device),
                        torch.ones((self.p[i].shape[1])).to(self.p[0].device),
                        self.n_iter,
                        self.tau,
                    )

                gk.append(sk)

        else:
            gk = [
                matching(p.cpu().detach().numpy()).float().to(self.p[0].device)
                for p in self.p
            ]

        m = self.reparamnet(gk)
        if x is not None and x.ndim == 1:
            x.unsqueeze_(0)

        if x is not None:
            return m(x)

        return m

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.reparamnet.output.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)

    def parameters(self, recurse: bool = True):
        return self.p

    def to(self, device):
        for p in self.p:
            p.data = p.data.to(device)

        return self


# expected loss
class RndLoss(nn.Module):
    def __init__(self, modela=None, criterion=None):
        super(RndLoss, self).__init__()

        self.criterion = criterion

        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb, input, target):
        random_l = torch.rand((1,)).to(input.device)

        for p1, p2 in zip(modelb.parameters(), self.modela.parameters()):
            p1.add_((random_l / (1 - random_l)) * p2.data)
            p1.mul_((1 - random_l))

        z = modelb(input)
        loss = self.criterion(z, target)

        return loss


# align model1 to model2
def permute_align(model1, model2, data_loader, epochs, device):
    """
    Align model1 to model2 using the rebasin network
    """
    # rebasin network for model1
    pi_model = RebasinNet(model1)
    pi_model.to(device)

    # mid point loss Eq 7
    criterion = RndLoss(model2, criterion=torch.nn.BCEWithLogitsLoss())

    # optimizer
    optimizer = torch.optim.AdamW(pi_model.p.parameters(), lr=0.1)

    for _ in range(epochs):
        # training step
        pi_model.train()
        for x, y in data_loader:
            y = y.unsqueeze(1).float()
            rebased_model = pi_model()
            loss = criterion(rebased_model, x.to(device), y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pi_model.eval()
    rebased_model = copy.deepcopy(pi_model())

    return rebased_model
