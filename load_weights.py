import json
import torch
import stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm
import argparse


def load_hf_checkpoint(dataset="pusht"):
    src = Path(swm.data.utils.get_cache_dir(), f"hf_{dataset}")
    out = Path(swm.data.utils.get_cache_dir(), dataset, "lewm_object.ckpt")

    cfg = json.loads((src / "config.json").read_text())
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(k):
        return MLP(
            input_dim=cfg[k]["input_dim"],
            output_dim=cfg[k]["output_dim"],
            hidden_dim=cfg[k]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    # Make sure format of config file is correct
    try:
        del cfg["predictor"]["_target_"]
        del cfg["action_encoder"]["_target_"]
    except KeyError:
        print("Config file is already in the correct format.")

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**cfg["predictor"]),
        action_encoder=Embedder(**cfg["action_encoder"]),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )

    # 1. Load the checkpoint from disk
    checkpoint = torch.load(src / "weights.pt", map_location="cpu")

    # Extract the actual state dict if it is wrapped in a checkpoint dictionary
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # 2. Create a new dictionary to hold the translated keys
    translated_state_dict = {}

    for key, tensor in state_dict.items():
        new_key = key

        # Target the mismatched ViT layers
        if "encoder.encoder.layer." in key:
            # Fix the base layer prefix
            new_key = new_key.replace("encoder.encoder.layer.", "encoder.layers.")

            # Translate Attention Q, K, V Projections
            new_key = new_key.replace("attention.attention.query", "attention.q_proj")
            new_key = new_key.replace("attention.attention.key", "attention.k_proj")
            new_key = new_key.replace("attention.attention.value", "attention.v_proj")

            # Translate Attention Output
            new_key = new_key.replace("attention.output.dense", "attention.o_proj")

            # Translate MLP / FeedForward layers
            new_key = new_key.replace("intermediate.dense", "mlp.fc1")
            new_key = new_key.replace("output.dense", "mlp.fc2")

        translated_state_dict[new_key] = tensor

    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)  # noqa: F841
    model.load_state_dict(translated_state_dict, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print("Successfully saved weights to ", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pusht")
    args = parser.parse_args()
    load_hf_checkpoint(args.dataset)
