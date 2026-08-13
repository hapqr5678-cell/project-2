import re

def fix_ae(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "s1 = (GRID + 2*1 - 3) // 2 + 1" in content:
        return
        
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
    
    # Inject dynamic s3 in class namespace so sequential can use it?
    # No, s3 is a local variable in __init__. We need to make sure the Sequential sees it.
    # Ah, the Sequential is built inside __init__, so it can use the local variable `s3`.
    
    # Fix forward pass
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

fix_ae('model/v0/ae.py')
