"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        proprio_encoder=None,
        num_views=1,
        enc_dim=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

        # Optional proprioception conditioning. When set, the low-dim proprio is
        # embedded and folded into the state embedding so LeWM's own latent
        # ``Z_lewm`` carries the low-dim state (needed for the downstream policy
        # feature). There is no proprio *readout* head: closed-loop control uses
        # a separate feature-readout bridge, not a proprio decode.
        self.proprio_encoder = proprio_encoder

        # Optional multi-view support. With ``num_views > 1`` each camera's cls
        # token gets a learned view embedding; the per-view tokens are then
        # concatenated and projected back to ``embed_dim`` by ``projector``
        # (whose ``input_dim`` must therefore be ``num_views * enc_dim``).
        self.num_views = num_views
        if num_views > 1:
            assert enc_dim is not None, "enc_dim is required when num_views > 1"
            self.view_embedding = nn.Parameter(torch.zeros(num_views, enc_dim))
            nn.init.normal_(self.view_embedding, std=0.02)
        else:
            self.view_embedding = None

    def encode(self, info):
        """Encode observations and actions into embeddings.

        Two encoder contracts are supported:

        * **Modular / injected encoder** (``encoder.encode_obs`` present): the
          encoder maps a raw observation dict ``info["obs"]`` (each entry shaped
          ``(B, T, ...)``) directly to a per-frame feature ``(B, T, D)``. Kept as
          a generic path; not used by the representation-learning LeWM recipe.
        * **Native ViT encoder** (default): ``info["pixels"]`` may be single-view
          ``(B, T, C, H, W)`` or multi-view ``(B, T, V, C, H, W)``; the cls token
          per view is projected by ``projector`` and (optionally) has proprio
          folded in from ``info["proprio"]``.
        """

        if hasattr(self.encoder, "encode_obs"):
            emb = self.encoder.encode_obs(info["obs"])  # (B, T, D)
            info["emb"] = emb
            if "action" in info:
                info["act_emb"] = self.action_encoder(info["action"])
            return info

        pixels = info['pixels'].float()
        b = pixels.size(0)

        if pixels.dim() == 6:  # multi-view: (B, T, V, C, H, W)
            v = pixels.size(2)
            flat = rearrange(pixels, "b t v c h w -> (b t v) c h w")
            output = self.encoder(flat, interpolate_pos_encoding=True)
            cls = output.last_hidden_state[:, 0]  # (B*T*V, enc_dim)
            cls = rearrange(cls, "(b t v) d -> b t v d", b=b, v=v)
            if self.view_embedding is not None:
                cls = cls + self.view_embedding.view(1, 1, v, -1)
            cls = rearrange(cls, "b t v d -> (b t) (v d)")  # concat views
        else:  # single-view: (B, T, C, H, W)
            flat = rearrange(pixels, "b t ... -> (b t) ...")
            output = self.encoder(flat, interpolate_pos_encoding=True)
            cls = output.last_hidden_state[:, 0]  # cls token

        emb = self.projector(cls)
        visual_emb = rearrange(emb, "(b t) d -> b t d", b=b)
        info["visual_emb"] = visual_emb

        # fold proprio into the state embedding
        if "proprio" in info and self.proprio_encoder is not None:
            prop_emb = self.proprio_encoder(info["proprio"])
            info["prop_emb"] = prop_emb
            info["emb"] = visual_emb + prop_emb
        else:
            info["emb"] = visual_emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
