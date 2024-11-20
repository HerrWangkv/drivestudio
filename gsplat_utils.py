import math
from typing import Optional

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
)

def prune(
    threshold: float,
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
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

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
        if distributed:
            assert (
                colors.dim() == 2
            ), "Distributed mode only supports per-Gaussian colors."
    else:
        raise NotImplementedError

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

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if packed:
            raise NotImplementedError
        else:
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass
    else:
        raise NotImplementedError

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        raise NotImplementedError

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass

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
    if colors.shape[-1] > channel_chunk:
        raise NotImplementedError
    else:
        gs_mask = contribution_above_threshold(
            threshold,
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )
    return gs_mask

def contribution_above_threshold(
    threshold: float,
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
):
    
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device
    mask = torch.zeros((N), device=device).bool()

    render_colors = torch.zeros(
        (C, image_height, image_width, colors.shape[-1]), device=device
    )
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
        channels = colors.shape[-1]
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
        mask[gs_ids[weights > threshold]] = True # Note that transmittances are always 1.0
    return mask