import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange
import torch
import torch.nn as nn
import numbers
import torch.nn.functional as F


# Color Normalization
class Aff_channel(nn.Module):
    def __init__(self, dim, channel_first = True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # x的形状: torch.Size([1, 16, 160, 160]), out_features: 16, hidden_features: 64, in_features: 16
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.out_features = out_features
        self.hidden_features = hidden_features
        self.in_features = in_features

    def forward(self, x):
        # x 的原始尺寸为: torch.Size([1, 16, 160, 160])
        n, c, h, w = x.size()  # 分别获取 batch_size, channels, height, width
        # 重塑 x 以适配全连接层: 形状从 [n, c, h, w] -> [n*h*w, c]
        x = x.view(-1, c)
        device = x.device
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # 将 x 的形状从 [n*h*w, c] 调整回 [n, c, h, w]
        x = x.view(n, self.out_features, h, w)
        return x


class PemBlock(nn.Module):      #dim = 16 prompt_dim = 128
    def __init__(self, dim, prompt_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=Aff_channel, init_values=1e-4, num_splits=4):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(dim, prompt_dim, kernel_size=1, stride=1, bias=False)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_conv = nn.Conv2d(prompt_dim + dim, dim, kernel_size=1, stride=1, bias=False)

        self.dim = dim
        self.num_splits = num_splits
        self.transformer_block = [  # 变换器块数组，每个块处理输入的一部分，通过自注意力机制进一步提取特征。
            TransformerBlock(dim=dim // num_splits, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for _ in range(num_splits)]

    def forward(self, x, prompt_param):
        B, C, H, W = x.shape
        x_begin = x
        x = x + self.pos_embed(x)                                                       # PEM里第一个3*3 Conv

        norm_x = x.flatten(2).transpose(1, 2)   # x.flatten(2)的作用是将高度(H)和宽度(W)维度展平，因此操作后的形状是(B, C, H*W)。.transpose(1, 2)将第二维(C)和第三维(H*W)交换，因此norm_x的形状变为(B, H*W, C)。
        norm_x = self.norm1(norm_x)             # PEM中的Norm
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)  # 执行后(B, C, H, W)
        x = x + self.drop_path(self.gamma_1 * self.conv2(
            self.attn(self.conv1(norm_x))))  # 1*1 Conv 5*5 Conv 1*1 Conv 配合残差， Norm在上面

        x_prompt_before = self.conv1x1(x)  # [b,prompt_dim,h,w]                         # 经过第一个残差后 处理成准备接收prompt状态
        x_prompt_weight = self.sigmoid(x_prompt_before)                                 # Sigmod
        prompt_param = F.interpolate(prompt_param, (H, W), mode="bilinear")             # AIPM处理成与PEM相同的维度
        #prompt_param = F.silu(prompt_param)                                            # Swish激活函数
        prompt = x_prompt_weight * prompt_param
        prompt = self.conv3x3(prompt)
        x_concat = torch.cat([x_begin, prompt], dim=1)  # (b,prompt_dim+dim,h,w)
        x_prompt_after = self.out_conv(x_concat)  # (b,dim,h,w) dim=64

        splits = torch.split(x_prompt_after, self.dim // self.num_splits, dim=1)
        transformered_splits = []
        for i, split in enumerate(splits):
            transformered_split = self.transformer_block[i](split)
            transformered_splits.append(transformered_split)
        x_result = torch.cat(transformered_splits, dim=1)
        return x_result


## RFAConv
class RFAConv(nn.Module):  # 基于Group Conv
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):  # (b,c,h,w)
        return self.body(x) # (b,c/2,h*2,w*2)


#########################################################################
# Adaptive Information Prompts Module//Chain-of-Thought Prompt  (AIPM)
class AIPMParam(nn.Module):
    def __init__(self, prompt_inch,  num_path=3):
        super(AIPMParam, self).__init__()

        # (128,32,32)->(64,64,64)->(32,128,128)
        self.chain_prompts = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=prompt_inch if idx == 0 else prompt_inch // (2 ** idx),
                out_channels=prompt_inch // (2 ** (idx + 1)),
                kernel_size=3, stride=2, padding=1
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        prompt_params = []
        prompt_params.append(x)
        for pe in self.chain_prompts:
            x = pe(x)
            prompt_params.append(x)
        return prompt_params


#######################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, drop=0.):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        #self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        #self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))                      #前半段
        # x torch.Size([1, 16, 160, 160])
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        x = x + self.drop_path(x_mlp)                         #后半段

        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        device = x.device
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        result = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight.to(device) + self.bias.to(device)
        return result


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        device = x.device
        self.project_in = self.project_in.to(device)
        self.dwconv = self.dwconv.to(device)
        self.project_out = self.project_out.to(device)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, dtype=torch.float32), requires_grad=True)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        self.qkv = self.qkv.to(device)
        self.qkv_dwconv = self.qkv_dwconv.to(device)
        self.project_out = self.project_out.to(device)
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature.to(device)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


#####################################
class net_backbone(nn.Module):
    


class IAPET(nn.Module):
    def __init__(self, in_dim=3,):
        super(IAPET, self).__init__()

        self.local_net = net_backbone(in_dim=in_dim)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)
        return img_high


if __name__ == "__main__":
    img = torch.Tensor(1, 3, 640, 640)
    net = IAPET()
    imghigh = net(img)
    print(imghigh.size())
    print('total parameters:', sum(param.numel() for param in net.parameters()))
