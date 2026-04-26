import torch
import os

path = "question_3_model/cyclegan_weights.pt"

if os.path.exists(path):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    print("\n=== Checkpoint structure ===")
    print(f"Type: {type(checkpoint)}")
    print(f"Keys: {list(checkpoint.keys())}")
    
    if "G_AB" in checkpoint:
        print(f"\n=== G_AB state dict (first 10 keys) ===")
        g_ab_keys = list(checkpoint["G_AB"].keys())
        print(f"Total keys: {len(g_ab_keys)}")
        for key in g_ab_keys[:10]:
            print(f"  {key}")
        
    if "G_BA" in checkpoint:
        print(f"\n=== G_BA state dict (first 10 keys) ===")
        g_ba_keys = list(checkpoint["G_BA"].keys())
        print(f"Total keys: {len(g_ba_keys)}")
        for key in g_ba_keys[:10]:
            print(f"  {key}")
else:
    print(f"File not found: {path}")
