import torch.nn as nn
import torch

patch_size = 7
pos_embed = torch.zeros((1,16, 8), dtype=int)


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def positional_encoding(patches, num_patches):
    pos_enc = pos_embed.repeat(patches.shape[0], 1, 1)
    return torch.cat((pos_enc,patches), dim=2)



class SelfAttention(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        head_dim = dim // num_head
        self.head_dim = head_dim
        self.query = nn.Linear(head_dim, head_dim)
        self.key = nn.Linear(head_dim, head_dim)
        self.value = nn.Linear(head_dim, head_dim)
        self.to_out = nn.Linear(head_dim, head_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = self.softmax(torch.matmul(q, k.transpose(-1,-2)) / (self.head_dim**0.5))
        out = torch.matmul(attention, v)

        out = self.to_out(out)


        return out

class Encoder(nn.Module):
    def __init__(self, *, dim=32, head_n=4):
        super().__init__()
        self.head_n = head_n
        self.head_dim = dim // head_n
        self.layer_norm = nn.LayerNorm(32)

        self.heads = nn.ModuleList([SelfAttention(dim, head_n) for _ in range(head_n)])
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):

        embedded_patches = x
        x = self.layer_norm(embedded_patches)
        results = []
        for i in range(self.head_n):
            # slice the tensor into head_n slices
            part = (x[:,:,i*self.head_dim:(i+1)*self.head_dim])
            results.append(self.heads[i](part))

        results = torch.cat(results, dim=2)

        out_mhsa = results + embedded_patches

        mlp_input = self.layer_norm(out_mhsa)

        out = out_mhsa + self.mlp(mlp_input)
        return out

class ViT(nn.Module):
    def __init__(self, image_size, linear_map_dim, patch_size, num_classes, head_n, mlp_dim=64):
        super().__init__()
        self.hidden_d = linear_map_dim

        self.chw = (1,28,28)
        self.n_patches = image_size // patch_size
        self.linear_mapper = nn.Linear(patch_size ** 2 , self.hidden_d)

        self.positional_encoding = positional_encoding
        self.embed_dim = self.hidden_d + 8
        self.encoder = Encoder(dim=self.embed_dim, head_n=head_n)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )
    def forward(self, img):
        patches = patchify(img, self.n_patches)
        tokens = self.linear_mapper(patches)
        embedded_patches = self.positional_encoding(tokens, self.n_patches)
        out = self.encoder(embedded_patches)
        out = self.classifier(out[:,0])
        return out

def load_pos_embed_mat():
    k=0
    for i in range(4):
        for j in range(4):
            pos_embed[0,k,i] = 1
            pos_embed[0,k,j+4] = 1
            k+=1