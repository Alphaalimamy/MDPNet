import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(Conv3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding))

    def forward(self, x):
        x = self.conv(x)
        return x

class LMAP(nn.Module):
    def __init__(self, in_dim):
        super(LMAP, self).__init__()

        # Learnable attention weights (alpha)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Pooling layers with different kernel sizes and strides
        self.pool1 = nn.AvgPool2d(kernel_size=16, stride=16)
        self.pool2 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Example larger kernel convolutions (if needed)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=5, padding=2)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)  # [1, 64, 64, 64]

        # Apply average pooling with different scales
        q1 = self.pool1(conv2)  # Pooled output at scale n1 [1, 64, 4, 4]
        q2 = self.pool2(conv2)  # Pooled output at scale n2 [1, 64, 8, 8]
        q3 = self.pool3(conv2)  # Pooled output at scale n3 [1, 64, 16, 16]
        q4 = self.pool4(conv2)  # Pooled output at scale n4  [1, 64, 32, 32]

        # Reshape pooled outputs into the required format
        pq1 = q1.view(q1.size(0), -1, q1.size(2) * q1.size(3))  # [1, 64, 16]
        pq2 = q2.view(q2.size(0), -1, q2.size(2) * q2.size(3))  # [1, 64, 64]
        pq3 = q3.view(q3.size(0), -1, q3.size(2) * q3.size(3))  # [1, 64, 256]
        pq4 = q4.view(q4.size(0), -1, q4.size(2) * q4.size(3))  # [1, 64, 1024]

        x_q = conv2.view(x.size(0), -1, x.size(2) * x.size(3)) * self.alpha  # [1, 64, 4096]

        # Concatenate the results along the last dimension
        proj_query = torch.cat((pq1, pq2, pq3, pq4, x_q), dim=2)  # [1, 64, 5456]
        return proj_query


class NMAP(nn.Module):
    def __init__(self, in_dim):
        super(NMAP, self).__init__()
        self.nmap = LMAP(in_dim)

    def forward(self, X):
        batch, C, h, w = X.size()
        proj_query = self.nmap(X)  # [1, 64, 5456] C x D

        # Compute energy matrix for attention
        energy = torch.bmm(proj_query, proj_query.permute(0, 2, 1))  # [1, 64, 64]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1) # [1, 64, 64] C x C

        proj_value = X.view(batch, C, -1)  # [1, 64, 4096] C x N

        # Perform weighted sum using attention
        out = torch.bmm(attention, proj_value)
        out = out.view(batch, C, h, w) + X
        return out




class DPFocus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.1):
        super(DPFocus, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels // 16 # Reduction ratio = 16

        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=in_channels)
        # R, S, T
        self.convS = Conv3x3(in_channels=in_channels, out_channels=self.out_channels)
        self.convR = Conv1x1(in_channels=in_channels, out_channels=self.out_channels)
        self.convT = Conv1x1(in_channels=in_channels, out_channels=self.out_channels)

        self.conv1x1_out = Conv1x1(in_channels=self.out_channels, out_channels=in_channels)

        # N-MAP in the DPFM
        self.nmap = NMAP(in_dim=self.out_channels)
        self.output = Conv1x1(in_channels=in_channels, out_channels=self.out_channels)

        # Dynamic Attention Mechanism: Learnable scaling factors
        self.dynamic_scaling = nn.Parameter(torch.tensor(1.0))  # Initialize with 0.5
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.size()

        conv1 = self.conv1(x)

        convS = self.convS(x)
        convT = self.convT(x)
        convR = self.convR(conv1)

        # Reshape R, S, T
        S_rshape = convS.reshape(b, self.convS.out_channels, h * w)   
        R_reshape = convR.view(b, self.convT.out_channels, h * w)  
        T_reshape = convT.view(b, self.convR.out_channels, h * w)

        U = R_reshape + S_rshape # Equation 7
        Uid = self.dynamic_scaling * F.softmax(U, dim=-1) # Equation  10
        T = F.softmax(T_reshape, dim=-1).transpose(1, 2)  

        Pid = torch.matmul(Uid, T) 
        polyp_target = self.non_linearity(Pid)
        polyp_target = self.attn_dropout(polyp_target) # [1, 4, 4]

        # Apply NMAP
        nmap = self.nmap(Uid.reshape(Uid.size(0), Uid.size(1), h, w)) 
        nmap_reshape = nmap.view(b, self.convR.out_channels, h * w)  # Reshape to [b, 32, 65536]

        nmap_polyp = nmap_reshape.mean(dim=2, keepdim=True) 
        augmented_polyp_target = torch.matmul(polyp_target, nmap_polyp) 
        enhanced_polyp = polyp_target + augmented_polyp_target  

        augmentedR = torch.matmul(enhanced_polyp.transpose(2, 1), nmap_reshape)  
        augmentedR = torch.sigmoid(augmentedR)  
        output = torch.matmul(enhanced_polyp, augmentedR).reshape(b, self.convR.out_channels, h, w) # [1, 4, 256, 256]

        conv1x1_out = self.conv1x1_out(output)
        output = self.output(conv1x1_out + x * self.dynamic_scaling)
        return F.sigmoid(output)

if __name__ == '__main__':
    img = torch.randn(1, 64, 256, 256)
    model = DPFocus(in_channels=64, out_channels=64)
    output = model(img)
    print(f"Output: {output.shape}")