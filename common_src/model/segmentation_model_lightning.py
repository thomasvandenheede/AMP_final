# common_src/model/segmentation_model_lightning.py
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import JaccardIndex # For mIoU

import sys
sys.path.append('/home/mramidi/final_assignment/common_src/model')
from segmentation_helpers import get_modified_deeplabv3 # Import the helper

class SegmentationLightningModel(L.LightningModule):
    def __init__(self, 
                 num_classes: int = 4, 
                 freeze_backbone_during_finetune: bool = True,
                 use_coco_weights_for_backbone: bool = True,
                 loss_fn_name: str = "CrossEntropyLoss", 
                 class_weights: list = None, 
                 optimizer_cfg: dict = None,
                 input_height: int = 520,  # <--- Add this
                 input_width: int = 520,
                 class_mapping: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=['hparams']) # Saves args like num_classes to self.hparams

        self.model = get_modified_deeplabv3(
            num_target_classes=self.hparams.num_classes,
            freeze_backbone=self.hparams.freeze_backbone_during_finetune,
            use_coco_weights=self.hparams.use_coco_weights_for_backbone
        )

        if loss_fn_name == "CrossEntropyLoss":
            ce_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
            self.criterion = nn.CrossEntropyLoss(weight=ce_weights) # Add ignore_index if needed
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")
        
        self.optimizer_cfg = optimizer_cfg if optimizer_cfg else {'lr': 1e-4, 'weight_decay': 1e-5} # Default fine-tuning optimizer

        # Metrics
        self.train_miou = JaccardIndex(task="multiclass", num_classes=self.hparams.num_classes, average="macro")
        self.val_miou = JaccardIndex(task="multiclass", num_classes=self.hparams.num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeepLabV3 returns an OrderedDict, {'out': output, 'aux': aux_output} if aux_loss is True
        # If aux_loss is False (as in get_modified_deeplabv3), it returns just the main output tensor
        output = self.model(x)
        return output['out'] if isinstance(output, dict) else output


    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        #images, masks = batch # Expects (image_tensor, mask_tensor)
        images = batch['image'] 
        masks = batch['mask']

        outputs = self(images) # Logits: (B, NumClasses, H, W)
        loss = self.criterion(outputs, masks) # Masks: (B, H, W) with class indices

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        preds = torch.argmax(outputs, dim=1)
        self.train_miou.update(preds, masks)
        self.log('train/mIoU', self.train_miou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        #images, masks = batch
        images = batch['image'] 
        masks = batch['mask']
        
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # Changed from validation/loss
        
        preds = torch.argmax(outputs, dim=1)
        self.val_miou.update(preds, masks)
        self.log('val/mIoU', self.val_miou, on_step=False, on_epoch=True, prog_bar=True) # Changed from validation/mIoU

    def configure_optimizers(self):
        # Only optimize parameters that require gradients (respects freeze_backbone)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        lr = self.hparams.optimizer_cfg.get('lr', 1e-4)
        weight_decay = self.hparams.optimizer_cfg.get('weight_decay', 1e-5)
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Optional: Add LR scheduler if configured
        # if self.hparams.get("lr_scheduler_cfg"):
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.lr_scheduler_cfg)
        #     return [optimizer], [scheduler]
        return optimizer