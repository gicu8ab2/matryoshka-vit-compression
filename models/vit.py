# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
import math

import einops


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
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

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_LearnPos(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def get_2d_sinusoid_encoding_table(height, width, d_hid):
    """2D Sinusoid position encoding table that preserves spatial structure"""
    def get_position_angle_vec(position, d_hid):
        return [position / (10000 ** (2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]

    # Split embedding dimension in half for x and y coordinates
    d_hid_half = d_hid // 2
    
    # Create position encodings for x and y coordinates
    pos_table = torch.zeros(height * width + 1, d_hid)  # +1 for CLS token
    
    # CLS token gets zero encoding (or could be learned)
    pos_table[0, :] = 0
    
    # Generate 2D positional encodings for patches
    for h in range(height):
        for w in range(width):
            patch_idx = h * width + w + 1  # +1 to account for CLS token at position 0
            
            # X coordinate encoding (first half of dimensions)
            x_encoding = get_position_angle_vec(w, d_hid_half)
            pos_table[patch_idx, :d_hid_half:2] = math.sqrt(2)*torch.sin(torch.tensor(x_encoding[::2]))  # even positions
            pos_table[patch_idx, 1:d_hid_half:2] = math.sqrt(2)*torch.cos(torch.tensor(x_encoding[1::2]))  # odd positions
            
            # Y coordinate encoding (second half of dimensions)  
            y_encoding = get_position_angle_vec(h, d_hid_half)
            pos_table[patch_idx, d_hid_half::2] = math.sqrt(2)*torch.sin(torch.tensor(y_encoding[::2]))  # even positions
            pos_table[patch_idx, d_hid_half+1::2] = math.sqrt(2)*torch.cos(torch.tensor(y_encoding[1::2]))  # odd positions

    return pos_table.unsqueeze(0)


class DualPatchNorm(nn.Module):
    def __init__(self, patch_size, dim, in_channels):
        super().__init__()
        hp, wp = patch_size
        patch_dim = hp * wp * in_channels
        self.hp = hp
        self.wp = wp
        #self.ln0 = nn.LayerNorm(patch_dim)
        self.ln0 = nn.LayerNorm(patch_dim, elementwise_affine=False)
        self.proj = nn.Linear(patch_dim, dim)
        #self.ln1 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        x = self.ln0(x)          # [b, num_patches, patch_dim]
        x = self.proj(x)         # [b, num_patches, dim]
        x = self.ln1(x)          # [b, num_patches, dim]
        return x


class ViTFixedPosDualPatchNorm(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Store patch grid dimensions for 2D positional encoding
        self.patch_height_count = image_height // patch_height
        self.patch_width_count = image_width // patch_width

        self.patch_formation = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_norm = DualPatchNorm((patch_height, patch_width), dim, channels)
    
        # Fixed 2D sinusoidal positional encoding that preserves spatial structure
        self.register_buffer('pos_embedding', get_2d_sinusoid_encoding_table(
            self.patch_height_count, self.patch_width_count, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_formation(img)
        x = self.patch_norm(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



class ViTFixedPos(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Store patch grid dimensions for 2D positional encoding
        self.patch_height_count = image_height // patch_height
        self.patch_width_count = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
             nn.Linear(patch_dim, dim),
        )
    
        # Fixed 2D sinusoidal positional encoding that preserves spatial structure
        self.register_buffer('pos_embedding', get_2d_sinusoid_encoding_table(
            self.patch_height_count, self.patch_width_count, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
