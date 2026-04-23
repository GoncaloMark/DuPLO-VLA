import h5py
import numpy as np
import torch

def save_to_h5(vlm_model, dataloader, filename="libero_features.h5"):
    vlm_model.eval()
    vlm_model.to("cuda")
    
    LAYER_INDICES = [8, 16, 24, 32]
    
    with h5py.File(filename, "w") as f:
        for i, batch in enumerate(dataloader):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = vlm_model(**batch['inputs'].to("cuda"), output_hidden_states=True)
                
                # Apply Norm and Filter Layers
                final_norm = vlm_model.model.norm
                # [Layers, Seq, Dim]
                sampled_hs = torch.stack([final_norm(outputs.hidden_states[idx]) for idx in LAYER_INDICES], dim=1)
                
            # Convert to float16 to save 50% space
            hs_np = sampled_hs.cpu().float().numpy().astype(np.float16)
            mask_np = batch['attention_mask'].cpu().numpy().astype(np.bool_)
            
            # Create a group for this frame/sample
            grp = f.create_group(f"sample_{i}")
            grp.create_dataset("hidden_states", data=hs_np, compression="lzf") # Fast compression
            grp.create_dataset("mask", data=mask_np)
            grp.create_dataset("episode_id", data=batch['episode_id'])