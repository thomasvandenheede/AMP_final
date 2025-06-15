# src/train.py
import os
import sys
import os.path as osp

# --- Path Setup ---
# Assumes train.py is in 'src/', and 'common_src' and 'config' are at the project root or accessible
# Adjust if your structure is different. One common way:
# current_dir = os.path.dirname(os.path.abspath(__file__)) # Gets directory of train.py
# project_root = os.path.abspath(os.path.join(current_dir, '..')) # Moves one level up to project root
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)
# For simplicity, using the previous relative path assumption, ensure it works for your structure
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, ListConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader
from PIL import Image # Needed for mask_transform lambda in segmentation if used directly here

# For CenterPoint (3D Detection)
from common_src.model.detector import CenterPoint # Ensure this path is correct
from common_src.dataset import ViewOfDelft, collate_vod_batch # Ensure collate_vod_batch is updated

# For Segmentation Fine-tuning
from common_src.model.segmentation_model_lightning import SegmentationLightningModel # Updated LightningModule

from torchvision import transforms # For image transforms
from vod.configuration import KittiLocations # For instantiating KittiLocations object for dataset

@hydra.main(version_base=None, config_path='../config', config_name='train_default') # Default config
def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    
    TRAINING_MODE = cfg.get("mode", "detection_pointpainting") 
    print(f"----- Running in Training Mode: {TRAINING_MODE} -----")
    print(f"Output directory: {cfg.output_dir}/{cfg.exp_id}")
    os.makedirs(osp.join(cfg.output_dir, cfg.exp_id), exist_ok=True)

    # Instantiate KittiLocations (needed by ViewOfDelft dataset)
    # cfg.kitti_locations should point to a config like config/kitti_locations/default.yaml
    kitti_locations_obj = hydra.utils.instantiate(cfg.kitti_locations, root_dir=cfg.data_root)


    if TRAINING_MODE == "segmentation_finetune":
        # --- Setup for Segmentation Fine-tuning ---
        
        # Frame ID loading (example, adapt to your split file or method)
        all_frame_ids_path = os.path.join(cfg.data_root, "lidar", "ImageSets", cfg.dataset_seg.get("all_frames_split_file", "train_val.txt"))
        if not os.path.exists(all_frame_ids_path):
            raise FileNotFoundError(f"Frame ID split file not found: {all_frame_ids_path}")
        with open(all_frame_ids_path, 'r') as f:
            all_frame_ids_sorted = sorted([line.strip() for line in f.readlines() if line.strip()])
        
        num_total_frames = len(all_frame_ids_sorted)
        split_ratio = cfg.dataset_seg.get("train_val_split_ratio", 0.8)
        num_train_frames = int(num_total_frames * split_ratio)
        
        train_frame_ids = all_frame_ids_sorted[:num_train_frames]
        val_frame_ids = all_frame_ids_sorted[num_train_frames:]
        print(f"Segmentation Fine-tuning: {len(train_frame_ids)} train frames, {len(val_frame_ids)} val frames.")

        # DeepLabV3 often expects specific input sizes, e.g., (520, 520)
        # These should be defined in config/model/segmentation_deeplab_model.yaml
        img_h = cfg.model.get('input_height', 520) 
        img_w = cfg.model.get('input_width', 520)  
        seg_target_size = (img_h, img_w)

        # The ViewOfDelft dataset in "segmentation_finetune" mode will handle transforms internally
        train_dataset = ViewOfDelft(
            data_root=cfg.data_root,
            split='train', # Or use custom split logic with train_frame_ids
            mode="segmentation_finetune",
            segmentation_target_size=seg_target_size,
            # Pass image_normalize_mean/std if they are top-level in cfg or from specific group
            image_normalize_mean=cfg.dataset_seg.get("image_normalize_mean", [0.485, 0.456, 0.406]),
            image_normalize_std=cfg.dataset_seg.get("image_normalize_std", [0.229, 0.224, 0.225]),
            cfg=cfg # Pass full cfg if needed for more specific settings
        )
        # Modify ViewOfDelft to accept frame_ids_list to use train_frame_ids
        # For now, assuming ViewOfDelft internally uses its 'split' param to get frames.
        # If you want to pass explicit frame_ids, ViewOfDelft.__init__ needs adjustment.

        val_dataset = ViewOfDelft(
            data_root=cfg.data_root,
            split='val', # Or use custom split logic with val_frame_ids
            mode="segmentation_finetune",
            segmentation_target_size=seg_target_size,
            image_normalize_mean=cfg.dataset_seg.get("image_normalize_mean", [0.485, 0.456, 0.406]),
            image_normalize_std=cfg.dataset_seg.get("image_normalize_std", [0.229, 0.224, 0.225]),
            cfg=cfg
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                      num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                                    num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        
        # cfg.model is from config/model/segmentation_deeplab_model.yaml
        model = hydra.utils.instantiate(cfg.model, optimizer_cfg=cfg.optimizer_seg) 
        checkpoint_monitor = cfg.get('checkpoint_monitor_seg', 'val/mIoU')
        checkpoint_mode = cfg.get('checkpoint_mode_seg', 'max')
        #wandb_config = cfg.wandb_logger_seg

    elif TRAINING_MODE == "detection_pointpainting":
        print("----- Running in 3D Detection (PointPainting) Training Mode -----")
        train_dataset = ViewOfDelft(
            data_root=cfg.data_root, split='train', mode="detection_pointpainting",
            load_image_for_pointpainting=True,
            segmentation_target_size=(cfg.model.image_segmentation_model.get('input_height',520), cfg.model.image_segmentation_model.get('input_width',520)), # Image size for DeepLabV3 inference
            image_normalize_mean=cfg.model.image_segmentation_model.get("image_normalize_mean", [0.485, 0.456, 0.406]),
            image_normalize_std=cfg.model.image_segmentation_model.get("image_normalize_std", [0.229, 0.224, 0.225]),
            cfg=cfg
        )
        val_dataset = ViewOfDelft(
            data_root=cfg.data_root, split='val', mode="detection_pointpainting",
            load_image_for_pointpainting=True,
            segmentation_target_size=(cfg.model.image_segmentation_model.get('input_height',520), cfg.model.image_segmentation_model.get('input_width',520)),
            image_normalize_mean=cfg.model.image_segmentation_model.get("image_normalize_mean", [0.485, 0.456, 0.406]),
            image_normalize_std=cfg.model.image_segmentation_model.get("image_normalize_std", [0.229, 0.224, 0.225]),
            cfg=cfg
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
                                      shuffle=True, collate_fn=collate_vod_batch, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.get('val_batch_size',1), num_workers=cfg.num_workers, # val_batch_size for detection
                                    shuffle=False, collate_fn=collate_vod_batch, pin_memory=True)
        
        # cfg.model is from config/model/centerpoint_model.yaml (or similar)
        model = CenterPoint(cfg.model) 
        checkpoint_monitor = cfg.get('checkpoint_monitor_det','validation/ROI/mAP')
        checkpoint_mode = cfg.get('checkpoint_mode_det','max')
        #wandb_config = cfg.wandb_logger_det

    else:
        raise ValueError(f"Unknown training mode: {TRAINING_MODE}")

    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(cfg.output_dir, cfg.exp_id, "checkpoints"),
            filename=f'ep{{epoch}}-{cfg.exp_id}',
            save_last=True, monitor=checkpoint_monitor, mode=checkpoint_mode,
            auto_insert_metric_name=False, save_top_k=cfg.save_top_model if cfg.save_top_model > 0 else 1,
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    wandb_save_dir = osp.join(cfg.output_dir, cfg.exp_id, 'wandb_logs')
    os.makedirs(wandb_save_dir, exist_ok=True)
    logger = WandbLogger(
        save_dir=osp.join(cfg.output_dir, 'wandb_logs'),
        project='amp', 
        name=cfg.exp_id,
        log_model=True,
    )
    
    model_to_watch = model.model if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module) else model
    if isinstance(model_to_watch, torch.nn.Module):
        logger.watch(model_to_watch, log='all', log_freq=100, log_graph=False)
    
    # Handle GPU configuration for Lightning Trainer
    devices_cfg = cfg.gpus
    if isinstance(devices_cfg, ListConfig):
        devices_cfg = list(devices_cfg) # Convert OmegaConf ListConfig to Python list
    elif isinstance(devices_cfg, int): # If it's an int, it means number of GPUs, or a single GPU ID
        if devices_cfg > 1 : # Multiple GPUs by count
            devices_cfg = devices_cfg 
        elif devices_cfg == 1: # Single GPU, should be a list like [0] or [specific_id]
             devices_cfg = [0] # Default to GPU 0 if 1 is specified as int
        elif devices_cfg == 0: # CPU
            devices_cfg = "auto" # Or 0 if lightning handles it as CPU
        # else: devices_cfg might be a single GPU ID like 0, then wrap in list
    # Ensure it's a list for multi-GPU or single specific GPU, or int for count.
    # For simplicity, assuming cfg.gpus is like [0] or [0,1] or an int for count.
    # Lightning handles devices="auto" as well.

    trainer = L.Trainer(
        logger=logger, log_every_n_steps=cfg.log_every, accelerator="gpu" if devices_cfg != "auto" and devices_cfg != 0 else "cpu",
        devices=devices_cfg if devices_cfg != "auto" else "auto", # Handles list of IDs or count
        check_val_every_n_epoch=cfg.val_every, strategy="auto", callbacks=callbacks,
        max_epochs=cfg.epochs,
        sync_batchnorm=cfg.sync_bn if (isinstance(devices_cfg,list) and len(devices_cfg)>1) or (isinstance(devices_cfg,int) and devices_cfg>1) else False,
        enable_model_summary=True,
    )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=cfg.get('checkpoint_path', None))
    wandb.finish()
    
if __name__ == '__main__':
    train()