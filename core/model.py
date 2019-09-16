from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet, densenet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM, NUM_CLS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


class ProposalNet(nn.Module):
    def __init__(self, init_feats=2048):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(init_feats, 128, 3, 1, 1)
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


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def visualize_regions(image, bb_coord, save_dir, save_name):
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = np.transpose(unorm(image).cpu().numpy(), [1, 2, 0])
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    colors = ['r', 'b', 'g', 'y']
    for bb_idx in range(CAT_NUM):
        width = bb_coord[bb_idx, -1] - bb_coord[bb_idx, 1]
        height = bb_coord[bb_idx, 2] - bb_coord[bb_idx, 0]
        rect = patches.Rectangle((bb_coord[bb_idx, 1], bb_coord[bb_idx, 0]), width, height,
                                 linewidth=1.5, edgecolor=colors[bb_idx], facecolor='none')
        ax.add_patch(rect)
    plt.savefig(os.path.join(save_dir, str(save_name) + '.png'))


class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        # self.pretrained_model = densenet.densenet201(pretrained=True)
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)  # 1 denotes the output size
        num_features = self.pretrained_model.num_features
        self.pretrained_model.fc = nn.Linear(num_features, NUM_CLS)  # in_features=512*4, out_features=200
        self.proposal_net = ProposalNet(num_features)
        self.topN = topN
        self.concat_net = nn.Linear(num_features * (CAT_NUM + 1), NUM_CLS)
        self.partcls_net = nn.Linear(num_features, NUM_CLS)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x, img_name=None, img_save_path=None):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()

        if img_save_path is not None:
            for i in range(batch):
                # img = np.transpose(x_pad[i].cpu().numpy(), [1, 2, 0])
                bb_coords = np.zeros((CAT_NUM, 4))
                for j in range(CAT_NUM):
                    bb_coords[j] = top_n_cdds[i][j, 1:5].astype(np.int).reshape(1, 4)
                visualize_regions(x_pad[i].clone(), bb_coords, img_save_path, img_name[i])

        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


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
