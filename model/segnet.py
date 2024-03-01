import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Encoder_res101, Encoder_res50
import utils.geom
import utils.vox
import utils.misc
import utils.basic


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        shared_out_channels = [256, 128, 64, 32, 16]
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      shared_out_channels[0],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(shared_out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels[0],
                      shared_out_channels[1],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(shared_out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels[1],
                      shared_out_channels[2],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(shared_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels[2],
                      shared_out_channels[3],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(shared_out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels[3],
                      shared_out_channels[4],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(shared_out_channels[4]),
            nn.ReLU(inplace=True),
        )
        self.final_head = nn.Conv2d(shared_out_channels[4],
                                    n_classes,
                                    kernel_size=1,
                                    padding=0)

    def forward(self, x, bev_flip_indices=None):

        x = self.segmentation_head(x)
        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(
                x[bev_flip2_index],
                [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        segmentation_output = self.final_head(x)
        return segmentation_output


class CrossAttention(nn.Module):
    def __init__(self, ):
        super(CrossAttention, self).__init__()
        self.all_head_size = 128

    def forward(self, query_layer, key_layer, value_layer):
        d_k = query_layer.size(-1)
        attention_scores = torch.matmul(query_layer / math.sqrt(d_k),
                                        key_layer.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(
            ln_out)

        return ln_out


class PoseQuery(nn.Module):
    def __init__(self, num_heads=4, dim=128):
        super(PoseQuery, self).__init__()
        self.num_heads = num_heads
        self.k_linear = nn.Linear(dim, dim, bias=True)
        self.v_linear = nn.Linear(dim, dim, bias=True)
        self.q_linear = nn.Sequential(
            nn.Linear(2, 32, bias=True),
            nn.Linear(32, 128, bias=True),
        )
        self.CrossAttention = CrossAttention()
        self.layer_norm_0 = LayerNormalization(dim)
        self.layer_norm_1 = LayerNormalization(dim)
        self.feedforward_0 = PositionwiseFeedForward(dim, 2048)
        self.feedforward_1 = PositionwiseFeedForward(dim, 2048)

    def forward(self, x, pose):
        B, C, X, Y = x.shape
        x = x.flatten(2).transpose(1, 2)
        B_, N, C = x.shape
        pose = pose.view(B_, 1, 2)
        q = self.q_linear(pose).reshape(B_, N, self.num_heads,
                                        C // self.num_heads).permute(
                                            0, 2, 1, 3)
        k = self.k_linear(x).reshape(B_, N, self.num_heads,
                                     C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(B_, N, self.num_heads,
                                     C // self.num_heads).permute(0, 2, 1, 3)
        atten_res = self.CrossAttention(q, k, v)
        q = q.transpose(1, 2).reshape(B_, N, C)
        atten_res = atten_res.view(B_, N, C)
        atten_res = self.layer_norm_1(atten_res +
                                      self.feedforward_0(atten_res))
        atten_res = atten_res.view(B, X, Y, C).permute(0, 3, 1, 2)
        return atten_res


class HybridBEV(nn.Module):
    def __init__(self,
                 Z,
                 Y,
                 X,
                 vox_util=None,
                 rand_flip=False,
                 latent_dim=128,
                 n_classes=5,
                 encoder_type="res101"):
        super(HybridBEV, self).__init__()
        assert (encoder_type in ["res101", "res50"])

        self.Z, self.Y, self.X = Z, Y, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456,
                                     0.406]).reshape(1, 3, 1,
                                                     1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224,
                                    0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)

        # pose_query_model
        self.pose_query_model = PoseQuery()

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=n_classes,
        )

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA,
                                             Z,
                                             Y,
                                             X,
                                             assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util,
                pose_angle):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        pose_angle: (B, 2)
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert (C == 3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 -
                      self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(
                rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(
                feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf / float(H)
        sx = Wf / float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # Geometry-based view transformer
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B * S, 1, 1)
        else:
            xyz_camA = None
        feat_bev_ = vox_util.unproject_image_to_mem(feat_camXs_,
                                                    utils.basic.matmul2(
                                                        featpix_T_cams_,
                                                        camXs_T_cam0_),
                                                    camXs_T_cam0_,
                                                    Z,
                                                    Y,
                                                    X,
                                                    xyz_camA=xyz_camA)
        feat_bev = __u(feat_bev_)
        mask_mems = (torch.abs(feat_bev) > 0).float()
        feat_bev = utils.basic.reduce_masked_mean(feat_bev, mask_mems, dim=1)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_bev[self.bev_flip1_index] = torch.flip(
                feat_bev[self.bev_flip1_index], [-1])
            feat_bev[self.bev_flip2_index] = torch.flip(
                feat_bev[self.bev_flip2_index], [-3])

        # POSE-Query module
        feat_bev = self.pose_query_model(feat_bev, pose_angle)

        # bev decoder
        seg_result = self.decoder(
            feat_bev, (self.bev_flip1_index,
                       self.bev_flip2_index) if self.rand_flip else None)

        return seg_result