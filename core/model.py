from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM
import copy

class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class FRE(nn.Module):
    def __init__(self):
        super(FRE, self).__init__()
        self.attn_hop = 2048
        self.spaConv = nn.Conv2d(2048, 196, 1)
        self.spaConv1 = nn.Conv2d(2048, 49, 1)
        self.fc1 = nn.Linear(196, 196, bias=False)
        self.fc2 = nn.Linear(196, 2048, bias=False)
        self.fc11 = nn.Linear(49, 49, bias=False)
        self.fc22 = nn.Linear(49, 2048, bias=False)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
       if x.shape[2] == 14:
          emb = self.spaConv(x)  # [bsz, 2400, 8, 8]#12,2400,14,14
          size = emb.size()  # [bsz, 2400, 8, 8]
          unfolded_embedding = emb.view(size[0], 196, -1)  # [bsz, 2400, 64]12,2400,196=14*14
          emb = unfolded_embedding.transpose(1, 2).contiguous()  # [bsz, 64, 2400]#12,196,2400
          unfolded_embedding = emb.view(-1, 196)  # [bsz*64, 2400]#2352=12*196,2400
          hbar_img = self.relu(self.fc1(self.drop(unfolded_embedding)))  # [bsz*64, attn-unit]#12,59,2400
          attn_img = self.fc2(hbar_img).view(size[0], size[2] * size[3], self.attn_hop)  # [bsz, 64, hop]#12,141600
          attn_img = torch.transpose(attn_img, 1, 2).contiguous()  # [bsz, hop, 64]#12,59,196
          out_img = torch.bmm(attn_img, emb).unsqueeze(3)  # [bsz, hop, 2400]#12,256,2400.1
          out_img = out_img.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
       else:
          emb = self.spaConv1(x)  # [bsz, 2400, 8, 8]#12,2400,14,14
          size = emb.size()  # [bsz, 2400, 8, 8]
          unfolded_embedding = emb.view(size[0], 49, -1)  # [bsz, 2400, 64]12,2400,196=14*14
          emb = unfolded_embedding.transpose(1, 2).contiguous()  # [bsz, 64, 2400]#12,196,2400
          unfolded_embedding = emb.view(-1, 49)  # [bsz*64, 2400]#2352=12*196,2400
          hbar_img = self.relu(self.fc11(self.drop(unfolded_embedding)))  # [bsz*64, attn-unit]#12,59,2400
          attn_img = self.fc22(hbar_img).view(size[0], size[2] * size[3], self.attn_hop)  # [bsz, 64, hop]#12,141600
          attn_img = torch.transpose(attn_img, 1, 2).contiguous()  # [bsz, hop, 64]#12,59,196
          out_img = torch.bmm(attn_img, emb).unsqueeze(3)  # [bsz, hop, 2400]#12,256,2400.1
          out_img = out_img.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
       return out_img # 8,1614
class GCN(nn.Module):

    def __init__(self, fpn_size=512, num_classes=59):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCN, self).__init__()
        self.proj_size = fpn_size
        total_num_selects = 4096
        num_joints = total_num_selects // 32
        A = torch.eye(num_joints) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size //4 , self.proj_size //4, 1)
        self.batch_norm = nn.BatchNorm1d(self.proj_size //4)

        self.conv_q1 = nn.Conv1d(self.proj_size //4, self.proj_size, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size //4, self.proj_size, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))


        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(nn.Linear(16384, 2048 * 5),
                                        nn.ELU(inplace=True),
                                        nn.Linear(2048 * 5 , num_classes))


        self.tanh = nn.Tanh()

    def forward(self, x):

        ### adaptive adjacency
        q1 = self.conv_q1(x).mean(1)
        k1 = self.conv_k1(x).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))

        A1 = self.adj1 + A1 * self.alpha1 

        ### graph convolution
        hs = self.conv1(x)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm(hs)
        
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs
class FREGNet(nn.Module):
    def __init__(self, topN=4):
        super(FREGNet, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 59)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Sequential(
            nn.Linear(16384, 2048 * (CAT_NUM + 1)),
            nn.ELU(inplace=True),
            nn.Linear(2048 * (CAT_NUM + 1), 59)
        )  # nn.Linear(2048 * (CAT_NUM + 1), 59)
        self.Liner = nn.Linear(128, 128)
        self.partcls_net = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 59)
        )
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int_)
        self.FRE = FRE()
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(2048 // 2),
            nn.Linear(2048 // 2, 448),
            nn.BatchNorm1d(448),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(448, 59),
        )
        self.avgpool = nn.AvgPool2d(7)
        self.conv_block1 = nn.Sequential(
            BasicConv(2048, 448, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(448, 2048 // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.GCNCombiner = GCN()

    def forward(self, x):
        batch = x.size(0)
        rpn_feature = self.pretrained_model(x)
        resnet_out = self.FRE(rpn_feature)  # 8,1024,1,1
        feature = self.avgpool(resnet_out)  # 16,2048,2,2
        feature11 = feature.view(feature.size(0),  feature.size(2) * feature.size(3), feature.size(1))
        feature1 = feature.view(feature.size(0), -1)
        feature1 = nn.Dropout(p=0.5)(feature1)  # 8,8192
        resnet_out = self.conv_block1(resnet_out).view(batch, -1)  # 8,1024,1,1
        resnet_out = self.classifier1(resnet_out)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int_)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int_)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        rpn_feature1 = self.pretrained_model(part_imgs.detach())
        part_features = self.ASM(rpn_feature1)
        part_features = self.avgpool(part_features)
        part_features = part_features.view(part_features.size(0), -1)
        part_features = nn.Dropout(p=0.5)(part_features)
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature1 = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature1, feature1], dim=1)

        concat_out1 = torch.cat([part_feature, feature11], dim=1).view(batch,-1 , 128)#.transpose(1,2).contiguous()#8,4096，4
        # print('123456', part_feature.shape, feature11.shape)
        Liner = self.Liner(concat_out1)
        # print('1111', Liner.shape)
        # concat_logits = self.GCNCombiner(feature,part_feature)
        concat_logits1 = self.concat_net(concat_out)
        concat_logits = self.GCNCombiner(Liner)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, concat_logits1, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss
