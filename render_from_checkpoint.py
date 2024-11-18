import os
import torch
import argparse
import imageio

import numpy as np

from torch import nn, Tensor
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from gsplat_utils import rasterization, prune 

from utils.misc import import_str
from utils.visualization import to8b, get_layout
from datasets.driving_dataset import DrivingDataset
from models.trainers.base import GSModelType
from models.gaussians.basics import dataclass_camera, dataclass_gs
from datasets.base.split_wrapper import SplitWrapper

class Dataset:
    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = os.path.join(
            self.data_cfg.data_root,
            f"{int(self.scene_idx):03d}"
        )
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"
        if os.path.exists(os.path.join(self.data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "ego_pose")))
        elif os.path.exists(os.path.join(self.data_path, "lidar_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "lidar_pose")))
        else:
            raise ValueError("Unable to determine the total number of frames. Neither 'ego_pose' nor 'lidar_pose' directories found.")
        # ---- find the number of synchronized frames ---- #
        if self.data_cfg.end_timestep == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        # to make sure the last timestep is included
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep
        self.build_pixel_source()
        self.layout = get_layout(self.data_cfg.dataset)
        self.aabb = self.get_aabb()
        self.full_image_set = self.build_split_wrapper()

    def build_pixel_source(self):
        self.data_cfg.pixel_source.load_sky_mask = False
        self.data_cfg.pixel_source.load_dynamic_mask = False
        self.data_cfg.pixel_source.load_objects = False
        self.data_cfg.pixel_source.load_smpl = False
        self.pixel_source = import_str(self.data_cfg.pixel_source.type)(
            self.data_cfg.dataset,
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.device,
        )
    
    @property
    def scene_idx(self) -> int:
        return self.data_cfg.scene_idx
    
    @property
    def num_img_timesteps(self) -> int:
        return self.pixel_source.num_timesteps
    
    def get_aabb(self) -> Tensor:
        aabb = self.pixel_source.get_aabb()
        return aabb
    
    def build_split_wrapper(self):
        full_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # cover all the images
            split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
            split="full",
        )
        return full_image_set
    
    def get_image(self, idx) -> dict:
        return self.full_image_set.get_image(idx, camera_downscale=1.0)
    
class Model:
    def __init__(self, cfg, dataset, threshold):
        self.threshold = threshold
        print(f"Pruning threshold: {self.threshold}")
        self.prune_or_not = True if self.threshold > 0 else False
        self.model_config = cfg.model
        self.render_cfg = cfg.trainer.render
        self.gaussian_optim_general_cfg = cfg.trainer.gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = cfg.trainer.gaussian_ctrl_general_cfg
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_timesteps = dataset.num_img_timesteps
        self.num_train_images = len(dataset.full_image_set)
        self.num_full_images = len(dataset.full_image_set)
        self.num_gs_points = 0
        self.max_num_per_frame = 0
        
        # init scene scale
        self._init_scene(scene_aabb=dataset.aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False

    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin

    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
                
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
        # msg = super().load_state_dict(state_dict, strict)
        
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)

    def process_camera(
        self,
        camera_infos,
        image_ids,
        novel_view=False,
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict
    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor # leave it here for future use
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
    
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=None        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def prune_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ):
        gs_mask = prune(
                threshold=self.threshold,
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze(),
                colors=gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
        return gs_mask

    def affine_transformation(
        self,
        rgb_blended,
        image_infos
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
        
    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ):
    
        def render_fn(opaticy_mask=None, return_info=False):
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity
        }        
        return results
    
    def render(self, idx):
        image_infos, camera_infos = self.dataset.get_image(idx)

        for k, v in image_infos.items():
            if isinstance(v, Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in camera_infos.items():
            if isinstance(v, Tensor):
                camera_infos[k] = v.cuda(non_blocking=True)
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)

        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )
        if self.prune_or_not:
            gs_mask = self.prune_gaussians(
                gs=gs,
                cam=processed_cam,
                near_plane=self.render_cfg.near_plane,
                far_plane=self.render_cfg.far_plane,
                radius_clip=self.render_cfg.get('radius_clip', 0.)
            )
        else:
            gs_mask = torch.ones_like(gs.means[:, 0], dtype=torch.bool)
        # render gaussians
        outputs = self.render_gaussians(
            gs=gs[gs_mask],
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        ret = self.affine_transformation(
            outputs["rgb_gaussians"], image_infos
        )
        return ret.cpu().numpy(), camera_infos["cam_name"], gs_mask

    def save_videos(
        self,
        save_pth: str,
        num_cams: int = 6,
        fps: int = 10,
    ):
        if self.num_timesteps == 1:  # it's an image
            writer = imageio.get_writer(save_pth, mode="I")
            return_frame_id = 0
        else:
            return_frame_id = self.num_timesteps // 2
            writer = imageio.get_writer(save_pth, mode="I", fps=fps)
        for i in trange(self.num_timesteps, desc="saving video", dynamic_ncols=True):
            merged_list = []
            rgbs = []
            cam_names = []
            mask = None
            for j in range(num_cams):
                with torch.no_grad():
                    rgb, cam_name, gs_mask = self.render(i*num_cams+j)
                    mask = gs_mask if mask is None else torch.logical_or(mask, gs_mask)
                    if self.num_gs_points == 0:
                        self.num_gs_points = len(mask)
                    else:
                        assert self.num_gs_points == len(mask), "Number of points in the mask is not consistent"
                rgbs.append(rgb)
                cam_names.append(cam_name)
            self.max_num_per_frame = max(self.max_num_per_frame, int(mask.sum()))
            tiled_img = self.dataset.layout(rgbs, cam_names)    
            merged_list.append(tiled_img)
            merged_frame = to8b(np.concatenate(merged_list, axis=0))
            writer.append_data(merged_frame)
        writer.close()
        del rgbs, cam_names
        print(f"Overall number of points            : {self.num_gs_points}")
        print(f"Maximum number of points per frame  : {self.max_num_per_frame}")

def parse_args():
    parser = argparse.ArgumentParser(description="Render from checkpoint")
    parser.add_argument('model', type=str, help='Path to the model checkpoint to load')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    model_path = args.model
    log_dir = os.path.dirname(model_path)
    config_path = os.path.join(log_dir, "config.yaml")
    print(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    dataset = Dataset(cfg.data)
    model = Model(cfg, dataset, threshold=0.1)
    model.resume_from_checkpoint(model_path)
    model.save_videos(os.path.join(log_dir, "rendered.mp4"))