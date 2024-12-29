import math
from typing import Optional

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal
import numpy as np
from plyfile import PlyData, PlyElement
from pytorch3d.ops import sample_farthest_points

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
)

def single_camera_importance(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    distributed: bool = False,
    ortho: bool = False,
    ) -> Tensor:

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        raise NotImplementedError

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        ortho=ortho,
    )

    if packed:
        raise NotImplementedError
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        raise NotImplementedError

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )

    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # print("rank", world_rank, "Before rasterize_to_pixels")
    importance = calculate_importance(
        means2d,
        conics,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    return importance

def calculate_importance(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    batch_per_iter: int = 100,
):
    
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    C, N = means2d.shape[:2]
    importance = torch.zeros(N, device=means2d.device)
    n_isects = len(flatten_ids)
    device = means2d.device
    mask = torch.zeros((N), device=device).bool()
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break
        try:
            from nerfacc import accumulate_along_rays, render_weight_from_alpha
        except ImportError:
            raise ImportError("Please install nerfacc package: pip install nerfacc")
        pixel_ids_x = pixel_ids % image_width
        pixel_ids_y = pixel_ids // image_width
        pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
        deltas = pixel_coords - means2d[camera_ids, gs_ids]  # [M, 2]
        c = conics[camera_ids, gs_ids]  # [M, 3]
        sigmas = (
            0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
            + c[:, 1] * deltas[:, 0] * deltas[:, 1]
        )  # [M]
        alphas = torch.clamp_max(
            opacities[camera_ids, gs_ids] * torch.exp(-sigmas), 0.999
        )

        indices = camera_ids * image_height * image_width + pixel_ids
        total_pixels = C * image_height * image_width

        weights, trans = render_weight_from_alpha(
            alphas, ray_indices=indices, n_rays=total_pixels
        )
        tmp = torch.zeros_like(importance)
        tmp.scatter_add_(0, gs_ids, weights) # due to index duplication and inplace operation
        importance += tmp # Note that transmittances are always 1.0
    return importance

class Gaussian:
    def __init__(self, gaussian):
        self._means = gaussian['_means']
        self._features_dc = gaussian['_features_dc']
        # self._features_rest = gaussian['_features_rest']
        self._opacities = gaussian['_opacities']
        self._scales = gaussian['_scales']
        self._quats = gaussian['_quats']
    @property
    def means(self):
        return self._means.detach().cpu().numpy()
    @property
    def features_dc(self):
        return self._features_dc.detach().cpu().numpy()
    # @property
    # def features_rest(self):
    #     return self._features_rest.detach().cpu().numpy()
    @property
    def opacities(self):
        return self._opacities.detach().cpu().numpy()
    @property
    def scales(self):
        return self._scales.detach().cpu().numpy()
    @property
    def quats(self):
        return self._quats.detach().cpu().numpy()
    def concat(self, other):
        self._means = torch.cat((self._means, other._means), dim=0)
        self._features_dc = torch.cat((self._features_dc, other._features_dc), dim=0)
        # self._features_rest = torch.cat((self._features_rest, other._features_rest), dim=0)
        self._opacities = torch.cat((self._opacities, other._opacities), dim=0)
        self._scales = torch.cat((self._scales, other._scales), dim=0)
        self._quats = torch.cat((self._quats, other._quats), dim=0)
        
    def __len__(self):
        assert self._means.shape[0] == self._features_dc.shape[0] == self._opacities.shape[0] == self._scales.shape[0] == self._quats.shape[0]# == self._features_rest.shape[0]
        return self._means.shape[0]

    def __getitem__(self, mask):
        return Gaussian(
            {
                '_means': self._means[mask],
                '_features_dc': self._features_dc[mask],
                # '_features_rest': self._features_rest[mask],
                '_opacities': self._opacities[mask],
                '_scales': self._scales[mask],
                '_quats': self._quats[mask],
            }
        )
    
    def __copy__(self):
        return Gaussian(
            {
                '_means': self._means.clone(),
                '_features_dc': self._features_dc,
                # '_features_rest': self._features_rest,
                '_opacities': self._opacities,
                '_scales': self._scales,
                '_quats': self._quats,
            }
        )

    def farthest_point_sample(self, num_points):
        _, indices = sample_farthest_points(self._means.unsqueeze(0), K=num_points, random_start_point=True)
        return self[indices[0].cpu().numpy()]

def standardize_rotation(rots):
    '''
    Different quaternions can represent the same orientation. This function standardizes the rotations.
    Args:
        rots: np.array of shape (N, 4)
    Returns:
        standard_rots: np.array of shape (N, 4)    
    '''
    permutations = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    symbols = np.array([[1, 1, 1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1]])
    rots_abs = np.abs(rots)
    max_idx = np.argmax(rots_abs, axis=1)
    standard_rots = rots[np.arange(len(rots))[:, None], permutations[max_idx]] * symbols[max_idx]
    tmp = 2 * (standard_rots[:,0] > 0)[:,None] - 1
    standard_rots *= tmp
    return standard_rots


def export_ply(gaussian, out_path):
    xyz = gaussian.means
    f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
    # f_rest = gaussian.features_rest.reshape((gaussian.features_rest.shape[0], -1))
    opacities = gaussian.opacities
    scales = gaussian.scales
    rotations = standardize_rotation(gaussian.quats)

    def construct_list_of_attributes(gaussian):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(gaussian.scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(gaussian.quats.shape[1]):
            l.append('rot_{}'.format(i))
        # for i in range(45):
        #     l.append('f_rest_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussian)]
    attribute_list = [xyz, f_dc, opacities, scales, rotations]#, f_rest]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(attribute_list, axis=1)
    # do not save 'features_extra' for ply
    # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(out_path)