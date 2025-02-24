
from linformer_pytorch import LinearAttentionHead, get_EF
import torch

E_proj = get_EF(64, 512)
# F_proj = get_EF(input_size, dim_k, method, dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj
F_proj = get_EF(64, 512)

model = LinearAttentionHead(
        dim=64, # Dim 2 of the input
        dropout=0.1, # Dropout of the P matrix
        E_proj=E_proj, 
        F_proj=F_proj, # The E and F layers
        causal_mask=True,
        full_attention=False, # Use Full Attention instead
        )
x = torch.randn(1, 512, 64)
y = model(x, x, x)
print(y) # (1, 512, 64)