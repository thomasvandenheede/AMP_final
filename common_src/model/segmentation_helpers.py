# common_src/model/segmentation_helpers.py
import torch
import torchvision.models.segmentation as tv_seg

def get_modified_deeplabv3(num_target_classes: int, freeze_backbone: bool = True, use_coco_weights: bool = True):
    """
    Loads a DeepLabV3 model with a ResNet50 backbone, pre-trained on COCO if specified,
    and replaces the classifier head to output `num_target_classes`.

    Args:
        num_target_classes (int): Number of classes for the new head.
        freeze_backbone (bool): If True, freezes all layers except the new classifier head.
        use_coco_weights (bool): If True, loads weights pre-trained on COCO. 
                                 Otherwise, initializes from scratch or with default ResNet50 weights.
    Returns:
        torch.nn.Module: The modified DeepLabV3 model.
    """
    weights = tv_seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if use_coco_weights else None
    model = tv_seg.deeplabv3_resnet50(weights=weights, progress=True, aux_loss=True) # aux_loss=False for simplicity

    if freeze_backbone:
        print("Freezing backbone of DeepLabV3.")
        for name, param in model.named_parameters():
            if not name.startswith('classifier.4'): # Keep the final Conv2d of the main classifier head trainable
                param.requires_grad = False
    
    # Replace the final Conv2d layer of the main classifier head
    # model.classifier is a ModuleList or Sequential, classifier[4] is often the final Conv2d
    old_classifier_final_layer = model.classifier[4]
    model.classifier[4] = torch.nn.Conv2d(
        old_classifier_final_layer.in_channels,
        num_target_classes,
        kernel_size=old_classifier_final_layer.kernel_size,
        stride=old_classifier_final_layer.stride,
        padding=old_classifier_final_layer.padding # Ensure padding is also copied if it's not (1,1)
    )

    # Handle auxiliary classifier if it exists and is used (default is aux_loss=None, so it might not be created or used)
    # If tv_seg.deeplabv3_resnet50(aux_loss=True) was used, model.aux_classifier would exist.
    # For simplicity with aux_loss=False, we don't strictly need to modify aux_classifier here,
    # but if it were enabled, its head would also need changing.
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        print("Modifying auxiliary classifier head.")
        if freeze_backbone: # Also freeze aux if backbone is frozen
            for name, param in model.aux_classifier.named_parameters():
                 if not name.startswith('4'): # Assuming aux_classifier[4] is its final conv
                    param.requires_grad = False
        
        old_aux_classifier_final_layer = model.aux_classifier[4]
        model.aux_classifier[4] = torch.nn.Conv2d(
            old_aux_classifier_final_layer.in_channels,
            num_target_classes,
            kernel_size=old_aux_classifier_final_layer.kernel_size,
            stride=old_aux_classifier_final_layer.stride,
            padding=old_aux_classifier_final_layer.padding
        )
    
    print(f"Modified DeepLabV3 head for {num_target_classes} classes. Backbone frozen: {freeze_backbone}")
    return model