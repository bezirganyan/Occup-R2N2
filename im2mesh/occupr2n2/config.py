import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.occupr2n2 import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    n_views = cfg['data']['n_views']
    z_dim = cfg['model']['z_dim']
    instance_loss = cfg['model']['instance_loss']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    n_classes = cfg['model']['n_classes']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    batch_size = cfg['training']['batch_size']


    #define the fully connected layer
    fc_size = 1024
    n_convilter = 128
    n_deconvfilter = 128
    n_gru_vox = 4
    conv3d_filter_shape = (n_convilter, n_deconvfilter, 3, 3, 3)
    h_shape = (batch_size, n_deconvfilter, n_gru_vox, n_gru_vox, n_gru_vox)
    c_dim = n_deconvfilter*(n_gru_vox**3)
    if encoder_latent == None:
        z_dim = 0
    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim, n_classes=n_classes, instance_loss=instance_loss,
        **decoder_kwargs
    )
    if encoder == "3dconvgru":
        encoder_kwargs = {"batch_size": batch_size,
                          "fc_size": fc_size,
                          "n_convilter": n_convilter,
                          "n_deconvfilter": n_deconvfilter,
                          "n_gru_vox": n_gru_vox,
                          "conv3d_filter_shape": conv3d_filter_shape,
                          "h_shape": h_shape,
                          "n_views": n_views
                         }

    if z_dim != 0 and encoder_latent != None:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs
        )
    else:
        encoder_latent = None

    print(len(dataset))
    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim,
            **encoder_kwargs
        )
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupR2N2Network(
        decoder, encoder, encoder_latent, p0_z, h_shape=h_shape, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    instance_loss = cfg['model']['instance_loss']

    if 'surface_loss_weight' in cfg['model']:
        surface_loss_weight = cfg['model']['surface_loss_weight']
    else:
        surface_loss_weight = 1.

    if ('loss_tolerance_episolon' in cfg['training']) and (0 in cfg['training']['loss_tolerance_episolon']):
        loss_tolerance_episolon = cfg['training']['loss_tolerance_episolon'][0]
    else:
        loss_tolerance_episolon = 0.

    if ('sign_lambda' in cfg['training']) and (0 in cfg['training']['sign_lambda']):
        sign_lambda = cfg['training']['sign_lambda'][0]
    else:
        sign_lambda = 0.

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        surface_loss_weight=surface_loss_weight,
        loss_tolerance_episolon=loss_tolerance_episolon,
        sign_lambda=sign_lambda,
        instance_loss=instance_loss
    )

    if 'loss_type' in cfg['training']:
        trainer.loss_type = cfg['training']['loss_type']
        print('loss type:', trainer.loss_type)

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
