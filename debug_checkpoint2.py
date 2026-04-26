import torch
import os

path = "question_3_model/cyclegan_weights.pt"

if os.path.exists(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    g_ab_state = checkpoint["G_AB"]
    print("=== Original G_AB keys (first 15) ===")
    orig_keys = list(g_ab_state.keys())
    print(f"Total: {len(orig_keys)}")
    for key in orig_keys[:15]:
        print(f"  {key}")
    
    print("\n=== After replace('module.model.', 'net.') (first 15) ===")
    clean = {k.replace("module.model.", "net."): v for k, v in g_ab_state.items()}
    new_keys = list(clean.keys())
    print(f"Total: {len(new_keys)}")
    for key in new_keys[:15]:
        print(f"  {key}")
    
    print("\n=== Expected ResNetGenerator keys ===")
    # Check what a fresh ResNetGenerator expects
    import sys
    sys.path.insert(0, '.')
    exec(open('app.py').read().split('# ─── Page Config')[0])  # Load only the model definitions
    
    model = ResNetGenerator()
    expected_keys = list(model.state_dict().keys())
    print(f"Total: {len(expected_keys)}")
    for key in expected_keys[:15]:
        print(f"  {key}")
    
    print("\n=== Comparison ===")
    print(f"Clean keys match expected: {set(new_keys) == set(expected_keys)}")
    if set(new_keys) != set(expected_keys):
        missing = set(expected_keys) - set(new_keys)
        unexpected = set(new_keys) - set(expected_keys)
        if missing:
            print(f"Missing from checkpoint: {list(missing)[:5]}")
        if unexpected:
            print(f"Unexpected in checkpoint: {list(unexpected)[:5]}")
else:
    print(f"File not found: {path}")
