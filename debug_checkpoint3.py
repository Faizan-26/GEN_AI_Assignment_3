import torch
import torch.nn as nn
import os

# Define ResNetBlock and ResNetGenerator (copied from app.py)
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, f=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, f, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(f), nn.ReLU(True),
            nn.Conv2d(f,   f*2, 3, 2, 1, bias=False), nn.InstanceNorm2d(f*2), nn.ReLU(True),
            nn.Conv2d(f*2, f*4, 3, 2, 1, bias=False), nn.InstanceNorm2d(f*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers.append(ResNetBlock(f * 4))
        layers += [
            nn.ConvTranspose2d(f*4, f*2, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f*2), nn.ReLU(True),
            nn.ConvTranspose2d(f*2, f,   3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f),   nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(f, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Load checkpoint
path = "question_3_model/cyclegan_weights.pt"
if os.path.exists(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    g_ab_state = checkpoint["G_AB"]
    
    print("=== Original G_AB keys (first 10) ===")
    orig_keys = sorted(g_ab_state.keys())[:10]
    for key in orig_keys:
        print(f"  {key}")
    
    print("\n=== After replace('module.model.', 'net.') (first 10) ===")
    clean = {k.replace("module.model.", "net."): v for k, v in g_ab_state.items()}
    new_keys = sorted(clean.keys())[:10]
    for key in new_keys:
        print(f"  {key}")
    
    print("\n=== Expected ResNetGenerator keys (first 10) ===")
    model = ResNetGenerator()
    expected_keys = sorted(model.state_dict().keys())[:10]
    for key in expected_keys:
        print(f"  {key}")
    
    print("\n=== Attempting to load ===")
    try:
        model.load_state_dict(clean)
        print("✓ SUCCESS: Model loaded without errors!")
    except RuntimeError as e:
        error_str = str(e)
        # Print first 500 chars of error
        print("✗ FAILED:")
        print(error_str[:500])
else:
    print(f"File not found: {path}")
