import torch
import torch.nn as nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# Code is adapted from: https://github.com/lucidrains/vit-pytorch
# More specifically: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py
# Accessed 2/8/24

# The ViT paper can be found here: https://arxiv.org/pdf/2010.11929

# Note: Patch size *MUST* be divisible by sequence length. I opt for ignoring the last input element within the architecture.
# TODO: I think you are supposed to pre-train these models as per paper..
# TODO: Read here: https://huggingface.co/docs/transformers/v4.27.0/model_doc/vit

# Modifications:
# I have altererd the basic ViT final layers to be a basic multi-task model.

# TODO: Make these a comparable param count to the FCN.

class ViT1D(nn.Module):
    def __init__(self, *, seq_len=3501, patch_size=20, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dim_head=64, dropout=0.2, emb_dropout=0.2):
        super().__init__()

        # Calculate number of patches based on the first 3500 elements (We exclude the last element here)
        num_patches = (seq_len - 1) // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Output head
        self.spg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 230)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.spg_head(cls_tokens)

class ViT1D_MultiTask(nn.Module):
    def __init__(self, *, seq_len=3501, patch_size=20, dim=128, depth=6, heads=6, mlp_dim=1024, channels=1, dim_head=32, dropout=0.2, emb_dropout=0.2):
        super().__init__()

        # Calculate number of patches based on the first 3500 elements (We exclude the last element here)
        num_patches = (seq_len-1) // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Multi-task output heads
        self.crysystem_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 7)
        )
        self.blt_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 7)
        )
        self.spg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 230)
        )
        self.composition_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 118)
        )

    def forward(self, series):

        # Slice off the last element
        series = series[:, :, :3500]

        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return {
            'crysystem': self.crysystem_head(cls_tokens),
            'blt': self.blt_head(cls_tokens),
            'spg': self.spg_head(cls_tokens),
            'composition': self.composition_head(cls_tokens)
        }
    
# Transformer classes:
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    