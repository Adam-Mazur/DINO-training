from src.transforms import DataAugmentationDINO
from src.model import DINOModel
from src.utils import set_seed
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import hydra


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    wandb_logger = WandbLogger(
        project="dino-training",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    run_ckpt_dir = Path(cfg.paths.model_checkpoint) / wandb_logger.experiment.id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = DataAugmentationDINO(
        cfg.model.transform.global_crops_scale,
        cfg.model.transform.local_crops_scale,
        cfg.model.transform.local_crops_number,
    )

    dataset = ImageFolder(cfg.paths.data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.train.batch_size_per_gpu,
        shuffle=True,
        num_workers=cfg.machine.num_workers,
        drop_last=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_ckpt_dir,
        filename="epoch{epoch:03d}",
        # keep all checkpoints
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.model.train.n_epochs,
        accelerator="gpu",
        devices=cfg.machine.num_gpus,
        num_nodes=cfg.machine.num_nodes,
        strategy="ddp",
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        gradient_clip_val=cfg.model.train.get("clip_grad", None),
        gradient_clip_algorithm="norm" if "clip_grad" in cfg.model.train else None,
        precision=cfg.machine.precision,
    )

    with trainer.init_module():
        model = DINOModel(
            output_dim=cfg.model.architecture.output_dim,
            use_bn_in_head=cfg.model.architecture.use_bn_in_head,
            norm_last_layer=cfg.model.architecture.norm_last_layer,
            model_name=cfg.model.architecture.model_name,
            patch_size=cfg.model.architecture.patch_size,
            drop_path_rate=cfg.model.architecture.drop_path_rate,
            lr=cfg.model.train.lr,
            min_lr=cfg.model.train.min_lr,
            batch_size_per_gpu=cfg.model.train.batch_size_per_gpu,
            warmup_epochs=cfg.model.train.warmup_epochs,
            weight_decay=cfg.model.train.weight_decay,
            weight_decay_end=cfg.model.train.weight_decay_end,
            warmup_teacher_temp=cfg.model.train.warmup_teacher_temp,
            teacher_temp=cfg.model.train.teacher_temp,
            warmup_teacher_temp_epochs=cfg.model.train.warmup_teacher_temp_epochs,
            student_temp=cfg.model.train.student_temp,
            center_momentum=cfg.model.train.center_momentum,
            local_crops_number=cfg.model.transform.local_crops_number,
            momentum_teacher=cfg.model.train.momentum_teacher,
            world_size=cfg.machine.num_gpus * cfg.machine.num_nodes,
            n_epochs=cfg.model.train.n_epochs,
            n_dataloader_steps=len(dataloader),
        )

    trainer.fit(model, dataloader, ckpt_path=cfg.paths.resume_from_checkpoint)


if __name__ == "__main__":
    main()
