"""Run Analysis using scripts from the PECAN Library: https://github.com/KrishnaswamyLab/PECAN"""

from pecan.functor import DiffusionCondensation


diffusion_condensation = DiffusionCondensation(callbacks=callbacks, kernel_fn=kernel_fn)
data = diffusion_condensation(X, epsilon=0.4)
