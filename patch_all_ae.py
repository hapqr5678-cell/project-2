import glob
import os

def fix_ae(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "s1 = (GRID + 2*1 - 3) // 2 + 1" in content:
        return False
        
    # Replace encoder Linear
    content = content.replace("nn.Linear(128 * 5 * 5, 256)", "nn.Linear(128 * s3 * s3, 256)")
    # Replace decoder Linear and Unflatten
    content = content.replace("nn.Linear(256, 128 * 5 * 5)", "nn.Linear(256, 128 * s3 * s3)")
    content = content.replace("nn.Unflatten(1, (128, 5, 5))", "nn.Unflatten(1, (128, s3, s3))")
    
    # Inject shape calculation
    init_str = "    def __init__(self, latent_dim=2):\n        super().__init__()"
    new_init_str = """    def __init__(self, latent_dim=2):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3"""
    content = content.replace(init_str, new_init_str)

    # Note: For some models latent_dim=16
    init_str_16 = "    def __init__(self, latent_dim=16):\n        super().__init__()"
    new_init_str_16 = """    def __init__(self, latent_dim=16):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3"""
    content = content.replace(init_str_16, new_init_str_16)

    # Some might not have latent_dim arg
    init_str_0 = "    def __init__(self):\n        super().__init__()"
    new_init_str_0 = """    def __init__(self):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3"""
    content = content.replace(init_str_0, new_init_str_0)
    
    # Fix forward pass (if standard)
    forward_str = """    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)"""
    new_forward_str = """    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1] or out.shape[-2] != x.shape[-2]:
            import torch.nn.functional as F
            out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return z, out"""
    content = content.replace(forward_str, new_forward_str)
    
    with open(path, 'w') as f:
        f.write(content)
    return True

for path in glob.glob('model/*/ae.py'):
    fixed = fix_ae(path)
    print(f"{path}: {'fixed' if fixed else 'skipped'}")

