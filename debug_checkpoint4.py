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
    
    print("=== Attempting to load with strict=False ===")
    model = ResNetGenerator()
    clean = {k.replace("module.model.", "net."): v for k, v in g_ab_state.items()}
    try:
        model.load_state_dict(clean, strict=False)
        print("✓ SUCCESS: Model loaded without errors!")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Model is in eval mode: {not model.training}")
    except Exception as e:
        print("✗ FAILED:")
        print(str(e)[:500])
else:
    print(f"File not found: {path}")
