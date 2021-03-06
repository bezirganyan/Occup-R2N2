import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, loss_type='cross_entropy',
                 surface_loss_weight=1.,
                 loss_tolerance_episolon=0.,
                 sign_lambda=0.,
                 instance_loss=False
                ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.loss_type = loss_type
        self.surface_loss_weight = surface_loss_weight
        self.loss_tolerance_episolon = loss_tolerance_episolon
        self.sign_lambda = sign_lambda
        self.instance_loss = instance_loss

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, self.instance_loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        #occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        #occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        #with torch.no_grad():
        #    elbo, rec_error, kl = self.model.compute_elbo(
        #        points, occ, inputs, **kwargs)

        target = data.get('points.point_lab').to(device)

#        eval_dict['rec_error'] = rec_error.mean().item()
#        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out, vote = self.model(points, inputs,
                               sample=self.eval_sample, **kwargs)

        instance_loss = True
        if instance_loss:
            centers = data.get('points.centers').to(device)
            vote_loss = (torch.max(vote.float() - centers.T.float(), torch.tensor([0.]).cuda())**2).sum()


        logits = p_out.logits
        if self.loss_type == 'cross_entropy':
            loss_i = F.cross_entropy(
                logits, target, reduction='none')
        occ_iou_np = (target >= 0).cpu().numpy()
        occ_iou_hat_np = p_out.probs.argmax(dim=1).cpu().numpy()
        #iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()

        #occ_iou_hat_np_04 = (p_out.probs >= 0.4).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou
        
        #loss = loss_i.sum(-1).mean()
        loss = loss_i.sum(-1).mean() + vote_loss * 1000

        eval_dict['loss'] = loss.cpu().numpy()

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out, vote = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = p_out.probs.argmax(dim=1).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r, vote = self.model(p, inputs, sample=self.eval_sample, **kwargs)

       # occ_hat = p_r.probs.reshape(batch_size,-1, *shape) # TODO - potential error here
       # occ_hat = occ_hat.max(dim=1)[0]
        occ_hat = p_r.probs.argmax(dim=1).cpu().numpy()
        voxels_out = occ_hat > 0

        #voxels_out = (occ_hat >= self.threshold).cpu().numpy()
        voxels_occ = data.get('voxels')
        voxels_out = voxels_out.reshape(voxels_occ.shape)
    

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))
            vis.visualize_voxels(
                voxels_occ[i], os.path.join(self.vis_dir, '%03d_gt.png' % i))

    def compute_loss(self, data, instance_loss=False, vote_weight=1000):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        #occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        target = data.get('points.point_lab').to(device)

        kwargs = {}
        c = self.model.encode_inputs(inputs)
        z = None
        loss = 0
        #if self.model.encoder_latent != None:
        #    q_z = self.model.infer_z(p, occ, c, **kwargs)
        #    z = q_z.rsample()

        #    # KL-divergence
        #    kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        #    loss = kl.mean()

        # General points
        if instance_loss:
            p_r, vote = self.model.decode(p, z, c, **kwargs)
        else:
            p_r = self.model.decode(p, z, c, **kwargs)
        logits = p_r.logits
        probs = p_r.probs
        vote_loss = 0
        if instance_loss:
            centers = data.get('points.centers').to(device)
            vote_loss = (torch.max(vote.float() - centers.T.float(), torch.tensor([0.]).cuda())**2).sum()

        if self.loss_type == 'cross_entropy':
            loss_i = F.cross_entropy(
                logits, target, reduction='none')
        elif self.loss_type == 'l2':
            logits = F.sigmoid(logits)
            loss_i = torch.pow((logits - occ), 2)
        elif self.loss_type == 'l1':
            logits = F.sigmoid(logits)
            loss_i = torch.abs(logits - occ)
        else:
            logits = F.sigmoid(logits)
            loss_i = F.binary_cross_entropy(logits, occ, reduction='none')

        if self.loss_tolerance_episolon != 0.:
            loss_i = torch.clamp(loss_i, min=self.loss_tolerance_episolon, max=100)

        if self.sign_lambda != 0.:
            w = 1. - self.sign_lambda * torch.sign(occ - 0.5) * torch.sign(probs - self.threshold)
            loss_i = loss_i * w

        if self.surface_loss_weight != 1.:
            w = ((occ > 0.) & (occ < 1.)).float()
            w = w * (self.surface_loss_weight - 1) + 1
            loss_i = loss_i * w

        #print('loss', loss, 'loss_i', loss_i.sum(-1).mean(), 'vote_loss', vote_loss)
        loss = loss + loss_i.sum(-1).mean() + vote_loss * vote_weight
        #loss = loss_i.sum(-1).mean()

        return loss

