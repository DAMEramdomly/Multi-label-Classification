import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from timm.models.layers import to_2tuple
from torch import einsum
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ConvGAU(nn.Module):
    def __init__(self, channels, query_key_channels=96,
                 expansion_factor=2., add_residual=True,
                 dropout=0.,):
        super().__init__()
        self.channels = channels
        hidden_channels = int(expansion_factor * channels)

        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Conv2d(channels, hidden_channels * 2, kernel_size=1),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Conv2d(channels, query_key_channels * 2, kernel_size=1),  # 输出分为q和k两部分
            nn.SiLU()
        )


        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self, x):

        v, gate = self.to_hidden(x).chunk(2, dim=1)

        Z = self.to_qk(x)
        q, k = Z.chunk(2, dim=1)

        B, C, H, W = q.shape
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, -1, H * W)

        sim = torch.bmm(q.permute(0, 2, 1), k)
        A = F.relu(sim) ** 2
        A = self.dropout(A)

        V = torch.bmm(v, A.permute(0, 2, 1))
        V = V.view(B, -1, H, W)
        V = V * gate

        if self.add_residual:
            x = x + self.to_out(V)

        return x

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output

class PAM_Module(nn.Module):
    """空间注意力模块"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, with_pos=True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=32)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        return x

class PositionEmbeddingLearned(nn.Module):
    """
    可学习的位置编码
    """

    def __init__(self, num_pos_feats=256, len_embedding=32):
        super().__init__()
        self.row_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.col_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos

class hybrid3(nn.Module):
	def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1], num_labels=4):
		self.filters = [32, 64, 128, 256]
		self.in_channels = 3
		self.inplanes = self.filters[0]
		self.num_labels = num_labels
		super(hybrid3, self).__init__()
		self.conv1 = nn.Conv2d(self.in_channels, self.filters[0], kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.filters[0])
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, self.filters[0], layers[0])
		self.layer2 = self._make_layer(block, self.filters[1], layers[1], stride=2)
		self.layer3 = self._make_layer(block, self.filters[2], layers[2], stride=2)
		self.layer4 = self._make_layer(block, self.filters[3], layers[3], stride=2)

		#self.MHSA = ConvGAU(channels=self.filters[3], query_key_channels=self.filters[3] * 2)
		self.MHSA2 = ScaledDotProductAttention(self.filters[3])
		self.MHSA3 = PAM_Module(self.filters[3])
		self.patch = PositionEmbeddingLearned(self.filters[3] // 2)


		self.avgpool = nn.AdaptiveAvgPool2d(1)

		self.fc_all = nn.Linear(self.filters[3] * block.expansion, self.num_labels)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		# modify the forward function
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# attention
		space_att = self.MHSA3(x)
		x_pos = self.patch(x)
		att = self.MHSA2(x_pos)
		final = att + space_att

		# global
		final = self.avgpool(att)
		final = final.view(x.size(0), -1)
		y = self.fc_all(final)


		return y


if __name__ == "__main__":

	dummy_input = torch.randn(4, 3, 256, 512)
	model = hybrid3()
	output = model(dummy_input)
	print(output.size())