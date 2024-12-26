import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, sn=False, norm='IN', activ='SiLU'):
        super(Conv2dBlock, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'BN':
            model += [nn.BatchNorm2d(out_channels)]
        elif norm == 'IN':
            model += [nn.InstanceNorm2d(out_channels)]

        if activ == 'LeakyReLU':
            model += [nn.LeakyReLU(negative_slope=0.2)]
        elif activ == 'ReLU':
            model += [nn.ReLU()]
        elif activ == 'SiLU':
            model += [nn.SiLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, channels, sn=False, norm='IN', activ='SiLU'):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2dBlock(channels, channels, 3, 1, 1, sn=sn, norm=norm, activ=activ)
        self.conv2 = Conv2dBlock(channels, channels, 3, 1, 1, sn=sn, norm=norm, activ='None')

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


class Correspondence(nn.Module):
    def __init__(self,
                in_channels_list,
                model_channels,
                num_res_blocks,
                channel_mult=(1, 2, 4, 8),
                use_fp16=False):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.channel_mult = channel_mult
        self.dtype = th.float16 if use_fp16 else th.float32
        ngf = 64
        self.softmax_alpha = 100
        self.eps = 1e-5

        self.proportion_prediction = nn.Sequential(
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, 1),
            nn.Sigmoid()
        )

        self.sd_enc = nn.Sequential(
            Conv2dBlock(in_channels_list[0]+2, ngf, 3, 1, 1),
            Conv2dBlock(ngf, ngf, 3, 1, 1),
            ResBlock(ngf * 1),
            ResBlock(ngf * 1),
            Conv2dBlock(ngf, ngf * 2, 3, 1, 1),
            ResBlock(ngf * 2),
            ResBlock(ngf * 2),
            Conv2dBlock(ngf * 2, ngf * 4, 3, 1, 1),
            ResBlock(ngf * 4),
            ResBlock(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 4, 1, 1, 0)
        )

        self.source_enc = nn.Sequential(
            Conv2dBlock(in_channels_list[1]+2, ngf, 3, 1, 1),
            Conv2dBlock(ngf, ngf, 3, 1, 1),
            ResBlock(ngf * 1),
            ResBlock(ngf * 1),
            Conv2dBlock(ngf, ngf * 2, 3, 1, 1),
            ResBlock(ngf * 2),
            ResBlock(ngf * 2),
            Conv2dBlock(ngf * 2, ngf * 4, 3, 1, 1),
            ResBlock(ngf * 4),
            ResBlock(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 4, 1, 1, 0)
        )

        self.ref_enc = nn.Sequential(
            Conv2dBlock(in_channels_list[2]+2, ngf, 3, 1, 1),
            Conv2dBlock(ngf, ngf, 3, 1, 1),
            ResBlock(ngf * 1),
            ResBlock(ngf * 1),
            Conv2dBlock(ngf, ngf * 2, 3, 1, 1),
            ResBlock(ngf * 2),
            ResBlock(ngf * 2),
            Conv2dBlock(ngf * 2, ngf * 4, 3, 1, 1),
            ResBlock(ngf * 4),
            ResBlock(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 4, 1, 1, 0)
        )

        proj_in_channels=in_channels_list[1]+in_channels_list[2]-19
        self.proj_convs = nn.ModuleList([nn.Conv2d(proj_in_channels, model_channels, 1, 1, 0)])

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * model_channels
                self.proj_convs.append(nn.Conv2d(proj_in_channels, ch, 1, 1, 0))
            if level != len(channel_mult) - 1:
                self.proj_convs.append(nn.Conv2d(proj_in_channels, ch, 1, 1, 0))

        print(f'The len of proj_convs:{len(self.proj_convs)}')
        print(self.proj_convs)

    def cal_correlation(self, fa, fb,seg_a,seg_b):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (fa.shape, fb.shape)
        n, c, h, w = fa.shape
        _, seg_c, _, _ = seg_a.shape
        # subtract mean
        fa = fa - th.mean(fa, dim=(2, 3), keepdim=True)
        fb = fb - th.mean(fb, dim=(2, 3), keepdim=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa = fa / (th.norm(fa, dim=1, keepdim=True) + self.eps)
        fb = fb / (th.norm(fb, dim=1, keepdim=True) + self.eps)

        seg_a = seg_a.view(n, seg_c, -1)
        seg_b = seg_b.view(n, seg_c, -1)

        energy_ab_T = th.bmm(fb.transpose(-2, -1), fa) * self.softmax_alpha
        mask_ab_T = th.bmm(seg_b.transpose(-2, -1), seg_a)
        energy_ab_T=energy_ab_T*mask_ab_T

        corr_ab_T = F.softmax(energy_ab_T, dim=1)  # n*HW*C @ n*C*HW -> n*HW*HW

        return corr_ab_T

    def add_coordinate(self,image):
        ins_feat = image  
        x_range = th.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
        y_range = th.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
        y, x = th.meshgrid(y_range, x_range) 
        y = y.expand([ins_feat.shape[0], 1, -1, -1])  
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = th.cat([x, y], 1)  
        output = th.cat([ins_feat, coord_feat], 1)  
        return output


    def forward(self, sd_x, source, ref, t,source_face_seg, LF,source_parsing,ref_parsing):
        # ref: z_ref 3x64x64  LF: unshuffle_ref 48x64x64
        sd_x = sd_x.type(self.dtype)
        source = source.type(self.dtype)
        ref = ref.type(self.dtype)
        source_face_seg = source_face_seg.type(self.dtype)
        t = t.type(self.dtype)
        LF = LF.type(self.dtype)
        source_parsing = source_parsing.type(self.dtype)
        ref_parsing = ref_parsing.type(self.dtype)

        source_parsing=F.interpolate(source_parsing,size=(sd_x.shape[2],sd_x.shape[3]))
        ref_parsing = F.interpolate(ref_parsing, size=(sd_x.shape[2], sd_x.shape[3]))


        all_LF = th.cat([ref, LF], dim=1)
        sd_x = sd_x * source_face_seg

        sd_x_add_coor = self.add_coordinate(sd_x)
        source_add_coor = self.add_coordinate(source)
        ref_add_coor = self.add_coordinate(all_LF)

        sd_x_add_coor_parsing=th.cat([sd_x_add_coor,source_parsing],dim=1)
        source_add_coor_parsing = th.cat([source_add_coor, source_parsing], dim=1)
        ref_add_coor_parsing = th.cat([ref_add_coor, ref_parsing], dim=1)
        # print(source_add_coor_parsing.shape)

        f_sd_x = self.sd_enc(sd_x_add_coor_parsing)
        f_source = self.source_enc(source_add_coor_parsing)
        f_ref = self.ref_enc(ref_add_coor_parsing)

        corr_sd_ref = self.cal_correlation(f_sd_x, f_ref,source_parsing,ref_parsing)
        corr_source_ref = self.cal_correlation(f_source, f_ref,source_parsing,ref_parsing)

        n, c, h, w = ref_add_coor.shape
        all_LF_warp_0 = th.bmm(ref_add_coor.reshape(n, c, h * w), corr_sd_ref)
        all_LF_warp_0 = all_LF_warp_0.reshape(n, c, h, w)

        all_LF_warp_1 = th.bmm(ref_add_coor.reshape(n, c, h * w), corr_source_ref)
        all_LF_warp_1 = all_LF_warp_1.reshape(n, c, h, w)

        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        proportion = self.proportion_prediction(t_emb)

        all_LF_warp = proportion[:, :, None, None] * all_LF_warp_0 + (
                    1.0 - proportion[:, :, None, None]) * all_LF_warp_1
        all_LF_warp = all_LF_warp * source_face_seg

        proj_input = th.cat([source, source_parsing, all_LF_warp[:, :-2, ::]], dim=1)
        outs = []
        for module in self.proj_convs:
            outs.append(module(proj_input))

        all_LF_warp_with_coor=th.cat([all_LF_warp[:, :-2, ::],all_LF_warp_0[:, -2:, ::],all_LF_warp_1[:, -2:, ::]],dim=1)
        return outs, all_LF_warp_with_coor



    def forward_test(self, sd_x, source, ref, t,source_face_seg, LF,source_parsing,ref_parsing):
        sd_x = sd_x.type(self.dtype)
        source = source.type(self.dtype)
        ref = ref.type(self.dtype)
        source_face_seg = source_face_seg.type(self.dtype)
        t = t.type(self.dtype)
        LF = LF.type(self.dtype)
        source_parsing = source_parsing.type(self.dtype)
        ref_parsing = ref_parsing.type(self.dtype)

        source_parsing = F.interpolate(source_parsing, size=(sd_x.shape[2], sd_x.shape[3]))
        ref_parsing = F.interpolate(ref_parsing, size=(sd_x.shape[2], sd_x.shape[3]))

        all_LF = th.cat([ref, LF], dim=1)
        sd_x = sd_x * source_face_seg

        sd_x_add_coor = self.add_coordinate(sd_x)
        source_add_coor = self.add_coordinate(source)
        ref_add_coor = self.add_coordinate(all_LF)

        sd_x_add_coor_parsing = th.cat([sd_x_add_coor, source_parsing], dim=1)
        source_add_coor_parsing = th.cat([source_add_coor, source_parsing], dim=1)
        ref_add_coor_parsing = th.cat([ref_add_coor, ref_parsing], dim=1)
        # print(source_add_coor_parsing.shape)

        f_sd_x = self.sd_enc(sd_x_add_coor_parsing)
        f_source = self.source_enc(source_add_coor_parsing)
        f_ref = self.ref_enc(ref_add_coor_parsing)

        corr_sd_ref = self.cal_correlation(f_sd_x, f_ref, source_parsing, ref_parsing)
        corr_source_ref = self.cal_correlation(f_source, f_ref, source_parsing, ref_parsing)

        n, c, h, w = ref_add_coor.shape
        all_LF_warp_0 = th.bmm(ref_add_coor.reshape(n, c, h * w), corr_sd_ref)
        all_LF_warp_0 = all_LF_warp_0.reshape(n, c, h, w)

        all_LF_warp_1 = th.bmm(ref_add_coor.reshape(n, c, h * w), corr_source_ref)
        all_LF_warp_1 = all_LF_warp_1.reshape(n, c, h, w)

        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        proportion = self.proportion_prediction(t_emb)

        # all_LF_warp = proportion[:, :, None, None] * all_LF_warp_0 + (
        #         1.0 - proportion[:, :, None, None]) * all_LF_warp_1
        # all_LF_warp = all_LF_warp * source_face_seg
        all_LF_warp = all_LF_warp_1* source_face_seg
        proj_input = th.cat([source, source_parsing, all_LF_warp[:, :-2, ::]], dim=1)
        outs = []
        for module in self.proj_convs:
            outs.append(module(proj_input))

        all_LF_warp_with_coor = th.cat([all_LF_warp[:, :-2, ::], all_LF_warp_0[:, -2:, ::], all_LF_warp_1[:, -2:, ::]],
                                       dim=1)
        data_dict={'LF_warp_0':F.pixel_shuffle(all_LF_warp_0[:, 3:3+3*16, ::]* source_face_seg,upscale_factor=4),
                   'LF_warp_1': F.pixel_shuffle(all_LF_warp_1[:, 3:3+3*16, ::]* source_face_seg,upscale_factor=4),
                   'LF_warp_2': F.pixel_shuffle(all_LF_warp[:, 3:3+3*16, ::]* source_face_seg,upscale_factor=4),
                   'proportion': proportion,
                   't':t
                   }
        return outs, all_LF_warp_with_coor,data_dict


