"""GRASP: Gradient RelAxed Stochastic Planner for stable_worldmodel.

Adapts GRASP pseudocode to work within the stable_worldmodel solver interface,
operating in the latent space of JEPA or PreJEPA world models.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gymnasium.spaces import Box


class GRASPSolver(torch.nn.Module):
    """GRASP planner that jointly optimizes actions and virtual states.

    Adapts to both JEPA and PreJEPA models by detecting the model type
    and using the appropriate encoding/prediction API.
    """

    def __init__(
        self,
        model,
        n_steps: int = 200,
        batch_size: int = 1,
        num_samples: int = 1,
        lr_s: float = 0.1,
        lr_a: float = 0.01,
        state_noise_scale: float = 0.01,
        gd_interval: int = 50,
        gd_opt_steps: int = 10,
        gd_lr: float = 0.05,
        sync_mode: str = "gd",
        cem_sync_samples: int = 64,
        cem_sync_topk: int = 8,
        cem_sync_var_scale: float = 1.0,
        cem_sync_var_min: float = 0.01,
        schedule_decay: bool = False,
        init_noise_scale: float = 0.1,
        min_noise_scale: float = 0.0,
        init_goal_weight: float = 1.0,
        min_goal_weight: float = 0.0,
        device: str | torch.device = "cuda",
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.lr_s = lr_s
        self.lr_a = lr_a
        self.state_noise_scale = state_noise_scale
        self.gd_interval = gd_interval
        self.gd_opt_steps = gd_opt_steps
        self.gd_lr = gd_lr
        self.sync_mode = sync_mode
        self.cem_sync_samples = cem_sync_samples
        self.cem_sync_topk = cem_sync_topk
        self.cem_sync_var_scale = cem_sync_var_scale
        self.cem_sync_var_min = cem_sync_var_min
        self.schedule_decay = schedule_decay
        self.init_noise_scale = init_noise_scale
        self.min_noise_scale = min_noise_scale
        self.init_goal_weight = init_goal_weight
        self.min_goal_weight = min_goal_weight
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None

        # Detect model type
        self._is_jepa = hasattr(model, 'encoder') and hasattr(model, 'action_encoder')
        self._is_prejepa = hasattr(model, 'backbone') and hasattr(model, 'extra_encoders')

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    # =========================================================================
    # JEPA model helpers (CLS-token based, simple (B, T, D) embeddings)
    # =========================================================================

    def _jepa_encode(self, pixels):
        """Encode pixels -> (B, D) using JEPA encoder + projector.
        pixels: (B, C, H, W) or (B, T, C, H, W)
        Returns: (B, D) embedding (flattened over time if T > 1, takes last)
        """
        if pixels.ndim == 4:
            # (B, C, H, W) - single frame per batch
            output = self.model.encoder(pixels, interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]  # CLS token
            emb = self.model.projector(pixels_emb)
            return emb  # (B, D)
        elif pixels.ndim == 5:
            # (B, T, C, H, W)
            b, t = pixels.shape[:2]
            pixels_flat = rearrange(pixels, "b t c h w -> (b t) c h w")
            output = self.model.encoder(pixels_flat, interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]
            emb = self.model.projector(pixels_emb)
            emb = rearrange(emb, "(b t) d -> b t d", b=b)
            return emb[:, -1, :]  # take last frame: (B, D)
        else:
            raise ValueError(f"Unexpected pixel tensor ndim: {pixels.ndim}")

    def _jepa_predict_step(self, emb, action):
        """Single-step prediction: emb (B, 1, D), action (B, 1, act_dim) -> (B, 1, D)."""
        act_emb = self.model.action_encoder(action)
        pred = self.model.predict(emb, act_emb)[:, -1:]
        return pred

    def _jepa_full_rollout(self, s_0, actions, history_size=None):
        """Full rollout: s_0 (B, 1, D), actions (B, T, act_dim) -> (B, T+1, D).

        Mirrors JEPA.rollout() but operates on pre-encoded embeddings.
        """
        if history_size is None:
            # Try to get from model, default to 1
            history_size = getattr(self.model, 'history_size', 1)
            # ARPredictor pos_embedding size
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'pos_embedding'):
                history_size = self.model.predictor.pos_embedding.shape[1]

        B, T, _ = actions.shape
        HS = history_size

        emb = s_0  # (B, 1, D)
        act = actions[:, :1, :]  # first action

        for t in range(T):
            act_seq = actions[:, :t + 1, :]
            act_emb = self.model.action_encoder(act_seq)

            # Truncate to history size
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]

            pred = self.model.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred], dim=1)

        return emb  # (B, T+1, D)

    # =========================================================================
    # PreJEPA model helpers (patch-based, (B, T, P, D) embeddings)
    # =========================================================================

    def _prejepa_encode(self, pixels):
        """Encode pixels -> (B, T, P, D) using PreJEPA backbone."""
        info = {"pixels": pixels}
        info = self.model.encode(info, pixels_key="pixels", target="embed", emb_keys=[])
        return info["embed"]  # (B, T, P, D)

    def _prejepa_predict_step(self, embed, action):
        """Single-step prediction for PreJEPA.
        embed: (B, 1, P, D), action: (B, 1, act_dim)
        Returns: (B, 1, P, D_pixel) predicted pixel embedding
        """
        # For PreJEPA, we need to incorporate action into the embedding
        if "action" in self.model.extra_encoders:
            n_patches = embed.shape[2]
            act_emb = self.model.extra_encoders["action"](action)  # (B, 1, act_D)
            act_tiled = act_emb.unsqueeze(2).expand(-1, -1, n_patches, -1)
            full_embed = torch.cat([embed, act_tiled], dim=-1)
        else:
            full_embed = embed

        pred = self.model.predict(full_embed)[:, -1:]
        # Split off the pixel part
        pixel_dim = embed.shape[-1]
        return pred[..., :pixel_dim]

    def _prejepa_full_rollout(self, s_0_embed, actions):
        """Full rollout for PreJEPA model.
        s_0_embed: (B, 1, P, D)
        actions: (B, T, act_dim)
        Returns: (B, T+1, P, D_pixel) trajectory of pixel embeddings
        """
        B, T, _ = actions.shape
        HS = self.model.history_size

        z = s_0_embed.clone()
        all_embeds = [s_0_embed]

        for t in range(T):
            act_seq = actions[:, :t + 1, :]
            if "action" in self.model.extra_encoders:
                n_patches = z.shape[2]
                act_emb = self.model.extra_encoders["action"](act_seq)
                act_tiled = act_emb.unsqueeze(2).expand(-1, -1, n_patches, -1)
                full_z = torch.cat([z, act_tiled], dim=-1)
            else:
                full_z = z

            pred = self.model.predict(full_z[:, -HS:])[:, -1:]
            pixel_dim = s_0_embed.shape[-1]
            pred_pixel = pred[..., :pixel_dim]
            all_embeds.append(pred_pixel)
            z = torch.cat([z, pred_pixel], dim=1)

        return torch.cat(all_embeds, dim=1)

    # =========================================================================
    # Generic latent-space helpers
    # =========================================================================

    def _encode_to_latent(self, pixels):
        """Encode pixels to a flat latent vector (B, D_flat) for GRASP optimization.
        pixels: (B, C, H, W) or (B, T, C, H, W)
        """
        with torch.no_grad():
            if self._is_jepa:
                return self._jepa_encode(pixels)  # (B, D)
            elif self._is_prejepa:
                if pixels.ndim == 4:
                    pixels = pixels.unsqueeze(1)  # add time dim
                emb = self._prejepa_encode(pixels)  # (B, T, P, D)
                return emb[:, -1].reshape(emb.shape[0], -1)  # (B, P*D)
            else:
                raise RuntimeError("Unknown model type")

    def _predict_step_latent(self, s_t, a_t):
        """Single step prediction in flat latent space.
        s_t: (B, D_flat), a_t: (B, 1, act_dim) -> (B, D_flat)
        """
        if self._is_jepa:
            s_t_emb = s_t.unsqueeze(1)  # (B, 1, D)
            pred = self._jepa_predict_step(s_t_emb, a_t)  # (B, 1, D)
            return pred[:, 0, :]
        elif self._is_prejepa:
            # Reshape flat -> (B, 1, P, D)
            P = self._n_patches
            D = s_t.shape[-1] // P
            s_t_emb = s_t.reshape(-1, 1, P, D)
            pred = self._prejepa_predict_step(s_t_emb, a_t)
            return pred[:, 0].reshape(-1, P * pred.shape[-1])
        else:
            raise RuntimeError("Unknown model type")

    def _full_rollout_latent(self, s_0_flat, actions):
        """Full rollout in flat latent space.
        s_0_flat: (B, D_flat), actions: (B, T, act_dim)
        Returns: (B, T+1, D_flat) trajectory
        """
        if self._is_jepa:
            s_0 = s_0_flat.unsqueeze(1)  # (B, 1, D)
            traj = self._jepa_full_rollout(s_0, actions)  # (B, T+1, D)
            return traj
        elif self._is_prejepa:
            P = self._n_patches
            D = s_0_flat.shape[-1] // P
            s_0_emb = s_0_flat.reshape(-1, 1, P, D)
            traj = self._prejepa_full_rollout(s_0_emb, actions)
            B, Tp1 = traj.shape[:2]
            return traj.reshape(B, Tp1, -1)
        else:
            raise RuntimeError("Unknown model type")

    # =========================================================================
    # CEM sync
    # =========================================================================

    def _cem_sync(self, s_0, g, a_opt, s_virtual, bs):
        """CEM-based sync: sample around current actions with adaptive per-timestep variance.

        Variance is set from the deviation between virtual states and actual
        rollout states — where the relaxation broke down, we explore more.
        A minimum variance floor ensures all actions get some exploration.
        """
        with torch.no_grad():
            a_mean = a_opt.detach().clone()  # (bs, T, act_dim)
            T = a_mean.shape[1]

            # Compute per-timestep variance from virtual state deviation
            traj = self._full_rollout_latent(s_0, a_mean)  # (bs, T+1, D)
            rollout_states = traj[:, 1:-1, :]  # (bs, T-1, D) — intermediate states
            virtual_states = s_virtual.detach()  # (bs, T-1, D)

            if virtual_states.shape[1] > 0:
                # Per-timestep deviation: (bs, T-1)
                deviation = ((rollout_states - virtual_states) ** 2).mean(dim=-1)
                # Expand to cover all T actions: pad first and last with mean deviation
                mean_dev = deviation.mean(dim=-1, keepdim=True)  # (bs, 1)
                deviation_full = torch.cat([mean_dev, deviation], dim=1)  # (bs, T)
                # Scale to variance
                var_t = self.cem_sync_var_scale * deviation_full + self.cem_sync_var_min
            else:
                # Horizon=1, no virtual states — use uniform variance
                terminal_gap = ((traj[:, -1] - g) ** 2).mean(dim=-1, keepdim=True)
                var_t = self.cem_sync_var_scale * terminal_gap + self.cem_sync_var_min

            # var_t shape: (bs, T) — broadcast over act_dim
            var_t = var_t.unsqueeze(-1)  # (bs, T, 1)

            for _ in range(self.gd_opt_steps):
                # Sample actions around mean with per-timestep variance
                noise = torch.randn(
                    bs, self.cem_sync_samples, T, a_mean.shape[-1],
                    device=a_mean.device, generator=self.torch_gen,
                )
                samples = a_mean.unsqueeze(1) + noise * var_t.unsqueeze(1).sqrt()
                # (bs, N, T, act_dim)

                # Evaluate each sample via full rollout
                costs = []
                for i in range(self.cem_sync_samples):
                    sample_traj = self._full_rollout_latent(s_0, samples[:, i])
                    s_T = sample_traj[:, -1, :]
                    cost = ((s_T - g) ** 2).sum(dim=-1)  # (bs,)
                    costs.append(cost)
                costs = torch.stack(costs, dim=1)  # (bs, N)

                # Select elites
                _, elite_idx = torch.topk(costs, self.cem_sync_topk, dim=1, largest=False)
                # Gather elite actions
                elite_actions = torch.gather(
                    samples, 1,
                    elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, a_mean.shape[-1])
                )  # (bs, topk, T, act_dim)

                # Update mean and variance from elites
                a_mean = elite_actions.mean(dim=1)  # (bs, T, act_dim)
                var_t = elite_actions.var(dim=1).mean(dim=-1, keepdim=True) + self.cem_sync_var_min
                # (bs, T, 1)

        return a_mean

    # =========================================================================
    # Main solve
    # =========================================================================

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Solve using GRASP: joint optimization of actions and virtual states."""
        start_time = time.time()

        device = self.device
        model = self.model
        total_envs = self._n_envs
        batch_size = self.batch_size if self.batch_size is not None else total_envs

        # For PreJEPA, figure out n_patches on first call
        if self._is_prejepa and not hasattr(self, '_n_patches'):
            # Will be set when we first encode
            self._n_patches = None

        all_top_actions = []

        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Extract batch info
            batch_info = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    batch_info[k] = v[start_idx:end_idx].to(device)
                elif isinstance(v, np.ndarray):
                    batch_info[k] = v[start_idx:end_idx]

            # Encode current observation and goal
            # Info tensors from WorldModelPolicy have shape (n_envs, T, C, H, W)
            # where T is history_len (usually 1)
            with torch.no_grad():
                obs_pixels = batch_info["pixels"]
                goal_pixels = batch_info["goal"]
                # Both should be (bs, T, C, H, W) at this point

                s_0 = self._encode_to_latent(obs_pixels)  # (bs, D_flat)
                g = self._encode_to_latent(goal_pixels)    # (bs, D_flat)

                # For PreJEPA, set _n_patches
                if self._is_prejepa and self._n_patches is None:
                    emb = self._prejepa_encode(obs_pixels)
                    self._n_patches = emb.shape[2]

            D_flat = s_0.shape[-1]
            T = self.horizon

            # Initialize actions
            if init_action is not None:
                a_init = init_action[start_idx:end_idx].to(device)
                remaining = T - a_init.shape[1]
                if remaining > 0:
                    a_init = torch.cat([
                        a_init,
                        torch.zeros(current_bs, remaining, self.action_dim, device=device)
                    ], dim=1)
            else:
                a_init = torch.zeros(current_bs, T, self.action_dim, device=device)

            # Initialize virtual states by linear interpolation
            num_virtual = T - 1
            if num_virtual > 0:
                t_vals = torch.linspace(0, 1, num_virtual + 2, device=device)[1:-1]
                t_vals = t_vals.view(1, -1, 1)
                s_virtual = s_0.unsqueeze(1) + t_vals * (g.unsqueeze(1) - s_0.unsqueeze(1))
            else:
                s_virtual = torch.empty(current_bs, 0, D_flat, device=device)

            s_virtual = s_virtual.clone().detach().requires_grad_(True)
            a_opt = a_init.clone().detach().requires_grad_(True)

            optimizer = torch.optim.Adam([
                {"params": s_virtual, "lr": self.lr_s},
                {"params": a_opt, "lr": self.lr_a},
            ])

            # Main GRASP optimization loop
            for k in range(self.n_steps):
                # Compute scheduled decay within current GD phase
                if self.schedule_decay and self.gd_interval > 0:
                    phase_step = k % self.gd_interval
                    decay_frac = phase_step / max(self.gd_interval - 1, 1)
                    cur_noise = self.init_noise_scale + (self.min_noise_scale - self.init_noise_scale) * decay_frac
                    cur_goal_weight = self.init_goal_weight + (self.min_goal_weight - self.init_goal_weight) * decay_frac
                else:
                    cur_noise = self.state_noise_scale
                    cur_goal_weight = 1.0

                optimizer.zero_grad()

                # Build full state sequence: [s_0, s_1...s_{T-1}, g]
                s_full = torch.cat([
                    s_0.unsqueeze(1),
                    s_virtual,
                    g.unsqueeze(1)
                ], dim=1)  # (bs, T+1, D_flat)

                loss_dyn = torch.tensor(0.0, device=device)
                loss_goal = torch.tensor(0.0, device=device)

                for t in range(T):
                    s_t = s_full[:, t].detach()  # stop-gradient on state input
                    a_t = a_opt[:, t:t + 1, :]   # (bs, 1, action_dim)

                    pred_next = self._predict_step_latent(s_t, a_t)  # (bs, D_flat)
                    s_next = s_full[:, t + 1]

                    loss_dyn = loss_dyn + F.mse_loss(pred_next, s_next)
                    loss_goal = loss_goal + F.mse_loss(pred_next, g.detach())

                loss = loss_dyn + cur_goal_weight * loss_goal
                loss.backward()
                optimizer.step()

                # Langevin-style noise on virtual states (scheduled or constant)
                if cur_noise > 0 and num_virtual > 0:
                    with torch.no_grad():
                        s_virtual.data += cur_noise * torch.randn_like(s_virtual)

                # Periodic full-rollout sync
                if self.gd_interval > 0 and (k + 1) % self.gd_interval == 0 and k > 0:
                    if self.sync_mode == "cem":
                        a_opt = self._cem_sync(
                            s_0, g.detach(), a_opt, s_virtual, current_bs,
                        ).requires_grad_(True)
                    else:
                        a_sync = a_opt.detach().clone().requires_grad_(True)
                        sync_optimizer = torch.optim.Adam([a_sync], lr=self.gd_lr)

                        for _ in range(self.gd_opt_steps):
                            sync_optimizer.zero_grad()
                            traj = self._full_rollout_latent(s_0, a_sync)
                            s_T = traj[:, -1, :]
                            sync_loss = F.mse_loss(s_T, g.detach())
                            sync_loss.backward()
                            sync_optimizer.step()

                        a_opt = a_sync.detach().clone()

                    # Rebuild optimizer with synced actions
                    a_opt = a_opt.detach().clone().requires_grad_(True)
                    optimizer = torch.optim.Adam([
                        {"params": s_virtual, "lr": self.lr_s},
                        {"params": a_opt, "lr": self.lr_a},
                    ])

            all_top_actions.append(a_opt.detach().cpu())

        outputs = {
            "actions": torch.cat(all_top_actions, dim=0),
            "cost": [],
        }

        elapsed = time.time() - start_time
        print(f"GRASPSolver.solve completed in {elapsed:.4f} seconds.")

        return outputs
