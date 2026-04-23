import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        # We don't keep the file open here because of multiprocessing issues
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())
        self.file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Open file once per worker process
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        grp = self.file[self.keys[idx]]
        
        # Load datasets into memory as numpy, then convert to torch
        # hidden_states: [x, L, D]
        hs = torch.from_numpy(grp["hidden_states"][:]).to(torch.bfloat16)
        mask = torch.from_numpy(grp["mask"][:])
        ep_id = int(grp["episode_id"][()])

        # Unbind into list of layers: [[L, D], [L, D]...]
        hs_list = list(hs.unbind(dim=0))

        return {
            "vlm_hidden_states": hs_list,
            "attention_mask": mask,
            "episode_id": ep_id
        }

def collate_fn(batch):
    num_layers = len(batch[0]['vlm_hidden_states'])
    
    # Stack each layer across the batch: List of tensors [B, L, D]
    collated_hs = [
        torch.stack([item['vlm_hidden_states'][l] for item in batch]) 
        for l in range(num_layers)
    ]
    
    masks = torch.stack([item['attention_mask'] for item in batch])
    episode_ids = [item['episode_id'] for item in batch]
    
    return {
        "vlm_hidden_states": collated_hs,
        "attention_mask": masks,
        "episode_ids": episode_ids
    }
