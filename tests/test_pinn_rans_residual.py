from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from models.train import pinn_rans_residual_loss


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PinnRansResidualTest(unittest.TestCase):
    class PoiseuilleField(torch.nn.Module if torch is not None else object):
        def __init__(self, pressure_gradient: float):
            super().__init__()
            self.pressure_gradient = pressure_gradient

        def forward(self, inputs):
            x, y = inputs[:, 0], inputs[:, 1]
            nu, rho = inputs[:, 2], inputs[:, 3]
            ux = 1.0 - y.square()
            uy = x * 0.0
            pressure = 10.0 + self.pressure_gradient * rho * nu * x
            nut = x * 0.0
            return torch.stack((ux, uy, pressure, nut), dim=1)

    def setUp(self):
        self.norm = SimpleNamespace(
            input_mean=np.zeros(5, dtype=np.float32),
            input_std=np.ones(5, dtype=np.float32),
            output_mean=np.zeros(4, dtype=np.float32),
            output_std=np.ones(4, dtype=np.float32),
        )
        x = torch.linspace(-0.8, 0.8, 7)
        y = torch.linspace(-0.8, 0.8, 7)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.inputs = torch.stack(
            (
                xx.flatten(),
                yy.flatten(),
                torch.full_like(xx.flatten(), 0.1),
                torch.full_like(xx.flatten(), 2.0),
                torch.ones_like(xx.flatten()),
            ),
            dim=1,
        )
        self.targets = ["Ux", "Uy", "p", "nut"]

    def residual(self, pressure_gradient):
        return pinn_rans_residual_loss(
            torch,
            self.PoiseuilleField(pressure_gradient),
            self.inputs,
            self.norm,
            self.targets,
            nu_index=2,
            rho_index=3,
            u_inf_index=4,
            weight=1.0,
            chord=1.0,
        )

    def test_balanced_pressure_and_viscous_terms_have_zero_residual(self):
        self.assertLess(self.residual(-2.0).item(), 1e-10)

    def test_missing_pressure_gradient_has_nonzero_momentum_residual(self):
        self.assertGreater(self.residual(0.0).item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
