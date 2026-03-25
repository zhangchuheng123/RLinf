# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Gaussian policy for SAC (inspired by dsrl_pi0).
# Reference: dsrl_pi0/jaxrl2/networks/learned_std_normal_policy.py

import torch
import torch.nn as nn


_LOG_2PI = float(torch.log(torch.tensor(2.0 * torch.pi)).item())


class TanhTransform(torch.distributions.Transform):
    """Tanh bijective transform with automatic log_abs_det_jacobian.

    Corresponds to distrax.Block(distrax.Tanh(), 1) in dsrl_pi0,
    which automatically computes the Jacobian correction.

    Transform: y = tanh(x)
    Jacobian: log|det J| = sum(log(1 - tanh^2(x_i))) = sum(log(1 - y_i^2))
    """

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self):
        super().__init__(cache_size=1)

    def __call__(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Clamp for numerical stability before atanh
        y = torch.clamp(y, -0.999999, 0.999999)
        return 0.5 * torch.log((1 + y) / (1 - y))

    def log_abs_det_jacobian(self, x, y):
        # Elementwise log|det J_i|; TransformedDistribution will handle event reduction.
        return torch.log(1 - y.pow(2) + 1e-7)


class SquashedNormal(torch.distributions.TransformedDistribution):
    """Squashed Gaussian distribution: N(mu, sigma) -> tanh -> [low, high].

    Corresponds to TanhMultivariateNormalDiag in dsrl_pi0.

    Combines:
    1. Diagonal multivariate Gaussian (MultivariateNormalDiag)
    2. Tanh transform
    3. Optional rescaling to [low, high]

    Key fix: Uses Independent to wrap univariate Normal into multivariate.
    """

    def __init__(self, loc, scale, low=-1.0, high=1.0, validate_args=False):
        # Create diagonal multivariate Gaussian.
        # Use Independent to wrap univariate Normal into multivariate.
        # loc.shape: [B, action_dim] -> batch_shape=[B], event_shape=[action_dim]
        normal = torch.distributions.Normal(loc, scale, validate_args=validate_args)
        base_dist = torch.distributions.Independent(
            normal, 1
        )  # Last dim becomes event_dim

        # Transform chain (applied inner to outer)
        transforms = []

        # Optional rescaling to [low, high]
        if low is not None and high is not None:
            # dsrl_pi0's rescale_from_tanh:
            # y = (x + 1) / 2 * (high - low) + low
            scale_factor = (high - low) / 2.0
            shift = (high + low) / 2.0

            class RescaleTransform(torch.distributions.Transform):
                domain = torch.distributions.constraints.interval(-1.0, 1.0)
                codomain = torch.distributions.constraints.interval(low, high)
                bijective = True
                sign = +1

                def __init__(self, scale, shift, cache_size=1):
                    super().__init__(cache_size=cache_size)
                    self.scale = scale
                    self.shift = shift

                def __call__(self, x):
                    return x * self.scale + self.shift

                def _inverse(self, y):
                    return (y - self.shift) / self.scale

                def log_abs_det_jacobian(self, x, y):
                    # Elementwise log|det J_i|; event reduction is handled upstream.
                    return torch.log(torch.abs(self.scale) * torch.ones_like(x))

            transforms.append(RescaleTransform(scale_factor, shift))

        # Tanh transform (applied last)
        transforms.append(TanhTransform())

        super().__init__(
            base_dist,
            torch.distributions.ComposeTransform(transforms),
            validate_args=validate_args,
        )

    @property
    def mean(self):
        # Note: the post-tanh mean is not simply tanh(base_mean).
        # Return base_mean for initialization reference.
        return self.base_dist.base_dist.loc

    @property
    def stddev(self):
        return self.base_dist.base_dist.scale

    def entropy(self):
        """Compute entropy: H = -E[log p(a)].

        Note: the transformed distribution has no closed-form entropy.
        Returns base_dist entropy as an approximation.
        """
        return self.base_dist.entropy()


class GaussianPolicy(nn.Module):
    """Gaussian policy network for SAC actor.

    Corresponds to LearnedStdTanhNormalPolicy in dsrl_pi0.

    Input: feature vector
    Output: SquashedNormal distribution (supports sampling and log_prob)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(128, 128, 128),  # Match dsrl_pi0
        log_std_init=-2.0,  # Initial log_std (smaller = more concentrated)
        low=None,  # Action lower bound
        high=None,  # Action upper bound
        action_horizon=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_horizon = action_horizon
        self.low = low
        self.high = high

        # Build shared MLP hidden layers
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim

        self.shared_net = nn.Sequential(*layers)

        # Mean and log_std output layers
        self.mean_layer = nn.Linear(in_dim, output_dim)
        self.log_std_layer = nn.Linear(in_dim, output_dim)

        # Initialize (reference: dsrl_pi0's default_init(1e-2))
        self._init_weights(log_std_init)

    def _init_weights(self, log_std_init):
        # Mean layer: small random init (matches dsrl_pi0's default_init(1e-2))
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)

        # log_std layer: small random init (matches dsrl_pi0).
        # dsrl_pi0 uses kernel_init=default_init(1e-2), so log_std starts near 0
        # (rather than being fixed to a specific value).
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        # Bias initialized to 0 so log_std starts at 0 (std = exp(0) = 1.0)
        nn.init.zeros_(self.log_std_layer.bias)

    def forward(self, features):
        """Forward pass returning a SquashedNormal distribution.

        Args:
            features: [B, input_dim] feature vector.

        Returns:
            distribution: SquashedNormal distribution.
        """
        # Shared feature extraction
        h = self.shared_net(features)

        # Compute mean and log_std
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        # Clamp log_std range (dsrl_pi0: clip(log_stds, log_std_min, log_std_max))
        log_std = torch.clamp(log_std, -20, 2)

        # Build distribution (equivalent to dsrl_pi0's TanhMultivariateNormalDiag)
        std = torch.exp(log_std)
        dist = SquashedNormal(mean, std, low=self.low, high=self.high)

        return dist

    def _compute_mean_log_std(self, features):
        h = self.shared_net(features)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def _log_prob_from_stats(self, mean, log_std, actions):
        if actions.dim() == 3:
            actions = actions[:, 0, :]

        if self.low is not None and self.high is not None:
            scale_factor = (self.high - self.low) / 2.0
            shift = (self.high + self.low) / 2.0
            y_t = (actions - shift) / scale_factor
            log_scale = torch.log(torch.abs(scale_factor) * torch.ones_like(y_t))
            scale_correction = torch.sum(log_scale, dim=-1)
        else:
            y_t = actions
            scale_correction = torch.zeros(actions.shape[0], device=actions.device)

        y_t = torch.clamp(y_t, -0.999999, 0.999999)
        x_t = 0.5 * torch.log((1 + y_t) / (1 - y_t))

        std = torch.exp(log_std)
        normal_log_prob = -0.5 * (((x_t - mean) / std) ** 2 + 2.0 * log_std + _LOG_2PI)
        normal_log_prob = torch.sum(normal_log_prob, dim=-1)

        tanh_correction = torch.sum(torch.log(1 - y_t.pow(2) + 1e-7), dim=-1)
        log_prob = normal_log_prob - tanh_correction - scale_correction
        return log_prob / float(actions.shape[-1])

    def _entropy_from_log_std(self, log_std):
        return torch.sum(0.5 * (1.0 + _LOG_2PI) + log_std, dim=-1)

    def sample(self, features, deterministic=False, return_stats=False):
        """Sample actions for inference.

        Uses CleanRL-style manual reparameterization to avoid issues with
        TransformedDistribution.rsample().

        Args:
            features: [B, input_dim]
            deterministic: Whether to use deterministic policy (mean).
            return_stats: Whether to return (mean, log_std) for monitoring.

        Returns:
            action: [B, action_horizon, output_dim]
            log_prob: [B]
            mean: [B, output_dim], optional
            log_std: [B, output_dim], optional
        """
        # Get mean and log_std
        mean, log_std = self._compute_mean_log_std(features)
        std = torch.exp(log_std)

        if deterministic:
            # Deterministic policy: use tanh(mean)
            action = torch.tanh(mean)
            # No log_prob needed for deterministic policy (no gradient update)
            log_prob = torch.zeros(features.shape[0], device=features.device)
        else:
            # Stochastic policy: CleanRL-style manual reparameterization
            # Steps:
            # 1. Sample from base Normal (reparameterization trick)
            # 2. Manually apply tanh
            # 3. Compute log_prob (with Jacobian correction)

            # Create base Normal distribution
            normal = torch.distributions.Normal(mean, std)

            # Sample from base distribution (pre-tanh space)
            x_t = normal.rsample()  # [B, output_dim], supports gradient flow

            # Manually apply tanh transform
            y_t = torch.tanh(x_t)  # [B, output_dim], in [-1, 1]

            # Optional: rescale to [low, high]
            if self.low is not None and self.high is not None:
                scale_factor = (self.high - self.low) / 2.0
                shift = (self.high + self.low) / 2.0
                action = y_t * scale_factor + shift
            else:
                action = y_t

        # Expand to action_horizon (after computing log_prob)
        # This matches the diffusion model's input format
        action = action.unsqueeze(1).repeat(
            1, self.action_horizon, 1
        )  # [B, action_horizon, output_dim]

        if not deterministic:
            log_prob = self._log_prob_from_stats(mean, log_std, action)

        if return_stats:
            return action, log_prob, mean, log_std
        else:
             return action, log_prob

    def evaluate_actions(self, features, actions, average_entropy=False):
        """Evaluate log_prob and entropy for given actions.

        Args:
            features: [B, input_dim]
            actions: [B, action_horizon, output_dim] or [B, output_dim]

        Returns:
            log_prob: [B]
            entropy: [B]
        """
        # If actions have action_horizon dimension, take the first timestep
        if actions.dim() == 3:
            actions = actions[:, 0, :]  # [B, output_dim]

        mean, log_std = self._compute_mean_log_std(features)
        log_prob = self._log_prob_from_stats(mean, log_std, actions)
        entropy = self._entropy_from_log_std(log_std)
        if average_entropy:
            entropy = entropy / float(actions.shape[-1])

        return log_prob, entropy
