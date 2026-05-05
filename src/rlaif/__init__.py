"""
RLAIF (Reinforcement Learning from AI Feedback) via DDPO
(Denoising Diffusion Policy Optimization) for the Neural Universal Machine.

Treats the 20-step Euler ODE as a Markov Decision Process where each step's
velocity prediction is the mean of a Gaussian policy. The GraphValidator's
5 Laws of Physics provide the terminal reward signal.
"""

from src.rlaif.naive_discretizer import naive_discretize
from src.rlaif.reward import compute_reward
from src.rlaif.rollout_buffer import RolloutBuffer
from src.rlaif.ppo import ppo_update
