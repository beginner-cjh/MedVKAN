import os
from os.path import join

import torch
from torch import device, nn
from torch._C import device
from torchinfo import summary
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UVKAN import get_u_vkan_from_plans

class nnUNetTrainerUVKAN(nnUNetTrainer):
    """ U-VKAN """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.early_stop_epoch = 350

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True
    ) -> nn.Module:

        model = get_u_vkan_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def configure_optimizers(self):  # 优化器和学习衰减
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler



    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0], [0.5, 0.5], [0.25, 0.25], [0.125, 0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
