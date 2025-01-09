import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
from .resnet import conv3x3

from qretinex import QRetinex, load_pretrained_model
# from qretinex.quaternion import hamilton_product  # if needed

# -----------------------------------------------------
# ASPP module unchanged
# -----------------------------------------------------
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate

        self.atrous_convolution = nn.Conv2d(
            inplanes, planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=rate,
            bias=False
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

# -----------------------------------------------------
# Main Network
# -----------------------------------------------------
class CrackNex(nn.Module):
    def __init__(self, backbone):
        super(CrackNex, self).__init__()

        # -----------------------------------------------------------------
        # 1) REPLACE OLD DecompNet with new QRetinex
        #    Freeze QRetinex weights so it is not trained with CrackNex.
        # -----------------------------------------------------------------
        self.QRetinex = load_pretrained_model(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for param in self.QRetinex.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # 2) Set up the two ResNet backbones:
        #    - rgb_backbone => 3-ch input
        #    - ref_backbone => 4-ch input
        # -------------------------------------------------
        rgb_backbone = resnet.__dict__[backbone](pretrained=True)
        ref_backbone = resnet.__dict__[backbone](pretrained=True)

        # Keep rgb_backbone as standard 3-channel
        self.rgb_layer0 = nn.Sequential(
            rgb_backbone.conv1,
            rgb_backbone.bn1,
            rgb_backbone.relu,
            rgb_backbone.maxpool
        )
        self.rgb_layer1 = rgb_backbone.layer1
        self.rgb_layer2 = rgb_backbone.layer2
        self.rgb_layer3 = rgb_backbone.layer3

        # Convert ref_backbone to accept 4-channel input
        old_weights = ref_backbone.conv1[0].weight.clone()  # shape (64, 3, 3, 3)
        new_conv = conv3x3(4, 64, stride=2)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weights
            new_conv.weight[:, 3, :, :] = 0.0
        ref_backbone.conv1[0] = new_conv  # replace the old conv

        # Freeze entire ref_backbone except the newly reinitialized conv1[0]
        for name, param in ref_backbone.named_parameters():
            param.requires_grad = False
        for param in ref_backbone.conv1[0].parameters():
            param.requires_grad = True

        self.ref_layer0 = nn.Sequential(
            ref_backbone.conv1,
            ref_backbone.bn1,
            ref_backbone.relu,
            ref_backbone.maxpool
        )
        self.ref_layer1 = ref_backbone.layer1
        self.ref_layer2 = ref_backbone.layer2
        self.ref_layer3 = ref_backbone.layer3

        # -----------------------------------------------------------------
        # Projection layers
        # -----------------------------------------------------------------
        self.proj_weight_FP = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.proj_weight_BP = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        # -----------------------------------------------------------------
        # ASPP modules
        # -----------------------------------------------------------------
        rates = [1, 6, 12, 18]
        self.aspp1 = ASPP_module(1024, 512, rate=rates[0])
        self.aspp2 = ASPP_module(1024, 512, rate=rates[1])
        self.aspp3 = ASPP_module(1024, 512, rate=rates[2])
        self.aspp4 = ASPP_module(1024, 512, rate=rates[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(2560, 1024, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.last_conv = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        )

        # -----------------------------------------------------------------
        # Special channel-split trick for rgb_layer2 (unchanged).
        # If you do something similar for the reflectance path, handle it here.
        # -----------------------------------------------------------------
        weight = self.rgb_layer2._modules['0'].conv1.weight.clone()
        self.rgb_layer2._modules['0'].conv1 = nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1), bias=False)
        self.rgb_layer2._modules['0'].conv1.weight.data[:, :256] = weight
        self.rgb_layer2._modules['0'].conv1.weight.data[:, 256:] = weight

        weight2 = self.rgb_layer2._modules['0'].downsample[0].weight.clone()
        self.rgb_layer2._modules['0'].downsample[0] = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(2,2), bias=False)
        self.rgb_layer2._modules['0'].downsample[0].weight.data[:, :256] = weight2
        self.rgb_layer2._modules['0'].downsample[0].weight.data[:, 256:] = weight2

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, img_s_list, hiseq_s_list, mask_s_list, img_q, hiseq_q, mask_q):
        h, w = img_q.shape[-2:]

        # Use QRetinex to decompose query into (Q1, Q2). We only use Q1 => reflectance_q
        with torch.no_grad():
            reflectance_q, _ = self.QRetinex(img_q)  # Q1=reflectance_q, Q2=illumination_q (ignored)

        # ----------------------------------------------
        # Feature maps of support images
        # ----------------------------------------------
        feature_s_list = []
        feature_s_ref_list = []
        feature_s_ref_lowlevel_list = []

        for k in range(len(img_s_list)):
            with torch.no_grad():
                # Decompose the support image
                reflectance_s, _ = self.QRetinex(img_s_list[k])

                s_0 = self.rgb_layer0(img_s_list[k])
                s_0_hiseq = self.rgb_layer0(hiseq_s_list[k])
                s_0 = self.rgb_layer1(s_0)
                s_0_hiseq = self.rgb_layer1(s_0_hiseq)

                s_0_ref = self.ref_layer0(reflectance_s)
                s_0_ref = self.ref_layer1(s_0_ref)
                s_0_lowlevel = s_0_ref

            s_0 = self.rgb_layer2(torch.cat([s_0, s_0_hiseq], dim=1))
            s_0 = self.rgb_layer3(s_0)

            s_0_ref = self.ref_layer2(s_0_ref)
            s_0_ref = self.ref_layer3(s_0_ref)

            s_0 = self.aspp_block(s_0, s_0_lowlevel)

            feature_s_list.append(s_0)
            feature_s_ref_list.append(s_0_ref)
            feature_s_ref_lowlevel_list.append(s_0_lowlevel)
            del s_0, s_0_hiseq, s_0_ref, s_0_lowlevel

        # ----------------------------------------------
        # Feature map of query image
        # ----------------------------------------------
        with torch.no_grad():
            q_0 = self.rgb_layer0(img_q)
            q_0_hiseq = self.rgb_layer0(hiseq_q)
            q_0 = self.rgb_layer1(q_0)
            q_0_hiseq = self.rgb_layer1(q_0_hiseq)

            q_0_ref = self.ref_layer0(reflectance_q)  # 4-ch input
            q_0_ref = self.ref_layer1(q_0_ref)
            q_0_lowlevel = q_0_ref

        q_0 = self.rgb_layer2(torch.cat([q_0, q_0_hiseq], dim=1))
        feature_q = self.rgb_layer3(q_0)
        feature_q = self.aspp_block(feature_q, q_0_lowlevel)

        # ----------------------------------------------
        # Generate foreground/background prototypes
        # ----------------------------------------------
        feature_fg_list = []
        feature_bg_list = []
        feature_ref_fg_list = []
        feature_ref_bg_list = []

        supp_out_ls = []
        supp_out_ref_ls = []

        for k in range(len(img_s_list)):
            # Original prototype
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]

            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            # Reflectance prototype
            feature_ref_fg = self.masked_average_pooling(feature_s_ref_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_ref_bg = self.masked_average_pooling(feature_s_ref_list[k], (mask_s_list[k] == 0).float())[None, :]

            feature_ref_fg_list.append(feature_ref_fg)
            feature_ref_bg_list.append(feature_ref_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(
                    feature_s_list[k],
                    feature_fg.squeeze(0)[..., None, None],
                    dim=1
                )
                supp_similarity_bg = F.cosine_similarity(
                    feature_s_list[k],
                    feature_bg.squeeze(0)[..., None, None],
                    dim=1
                )
                supp_out = torch.cat(
                    (supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1
                ) * 10.0

                supp_out = F.interpolate(
                    supp_out, size=(h, w),
                    mode="bilinear", align_corners=True
                )
                supp_out_ls.append(supp_out)

                # Reflectance support
                supp_similarity_ref_fg = F.cosine_similarity(
                    feature_s_ref_list[k],
                    feature_ref_fg.squeeze(0)[..., None, None],
                    dim=1
                )
                supp_similarity_ref_bg = F.cosine_similarity(
                    feature_s_ref_list[k],
                    feature_ref_bg.squeeze(0)[..., None, None],
                    dim=1
                )
                supp_out_ref = torch.cat(
                    (supp_similarity_ref_bg[:, None, ...], supp_similarity_ref_fg[:, None, ...]), dim=1
                ) * 10.0

                supp_out_ref = F.interpolate(
                    supp_out_ref, size=(h, w),
                    mode="bilinear", align_corners=True
                )
                supp_out_ref_ls.append(supp_out_ref)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        ref_FP = torch.mean(torch.cat(feature_ref_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        ref_BP = torch.mean(torch.cat(feature_ref_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # Fuse two prototypes => updated features & prototypes
        cate_FP = torch.cat((FP, ref_FP), dim=1)
        cate_BP = torch.cat((BP, ref_BP), dim=1)

        normalized_FP = self.z_score_norm(cate_FP)
        normalized_BP = self.z_score_norm(cate_BP)

        weights_FP = self.proj_weight_FP(normalized_FP)
        weights_BP = self.proj_weight_BP(normalized_BP)

        FP = torch.mul((1 + weights_FP), FP)
        feature_q = torch.mul((1 + weights_FP), feature_q)
        BP = torch.mul((1 + weights_BP), BP)

        # measure the similarity
        out_0 = self.similarity_func(feature_q, FP, BP)
        out_1, out_2 = self.SSP_module(feature_q, out_0, FP, BP)

        out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
        out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_2, out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat(
                (self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1
            ) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)
            supp_out_ref = torch.cat(supp_out_ref_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)
            out_ls.append(supp_out_ref)

        return out_ls

    # ---------------------------------------------------------------------
    # ASPP block (unchanged)
    # ---------------------------------------------------------------------
    def aspp_block(self, x, low_level_features):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.interpolate(
            x,
            size=(low_level_features.size()[-2], low_level_features.size()[-1]),
            mode='bilinear',
            align_corners=True
        )

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        return x

    # ---------------------------------------------------------------------
    # Z-score normalization
    # ---------------------------------------------------------------------
    def z_score_norm(self, tensor):
        mu = torch.mean(tensor, dim=(1), keepdim=True)
        sd = torch.std(tensor, dim=(1), keepdim=True)
        normalized_tensor = (tensor - mu) / (sd + 1e-5)
        return normalized_tensor

    # ---------------------------------------------------------------------
    # Self-Support Prototype module
    # ---------------------------------------------------------------------
    def SSP_module(self, feature_q, out_0, FP, BP):
        # 1st iteration
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        # 2nd iteration
        SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)
        FP_2 = FP * 0.5 + SSFP_2 * 0.5
        BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

        FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
        BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3

        out_2 = self.similarity_func(feature_q, FP_2, BP_2)
        out_2 = out_2 * 0.7 + out_1 * 0.3

        return out_1, out_2

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]

        fg_ls, bg_ls = [], []
        fg_local_ls, bg_local_ls = [], []
        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(1024, -1)
            f_h, f_w = feature_q[epi].shape[-2:]

            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]

            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)

            cur_feat_norm_t = cur_feat_norm.t()  # (N3, 1024)
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())  # (N3, 1024)
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())  # (N3, 1024)

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat(
            (similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1
        ) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(
            mask.unsqueeze(1),
            size=feature.shape[-2:],
            mode='bilinear',
            align_corners=True
        )
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
