from src.utils import trunc_normal_, cosine_scheduler, get_params_groups
from src.vision_transformer import vit_tiny, vit_small, vit_base
import torchvision.models as models
import pytorch_lightning as pl
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class DINOModel(pl.LightningModule):
    def __init__(
        self,
        output_dim: int,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        model_name: str = "resnet18",
        patch_size: int | None = None,
        drop_path_rate: float = 0.0,
        lr: float = 0.0005,
        min_lr: float = 0.0001,
        batch_size_per_gpu: int = 64,
        warmup_epochs: int = 10,
        weight_decay: float = 0.04,
        weight_decay_end: float = 0.4,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        local_crops_number: int = 8,
        momentum_teacher: float = 0.996,
        world_size: int = 1,
        n_epochs: int = 100,
        n_dataloader_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = 2 + local_crops_number
        self.n_dataloader_steps = n_dataloader_steps

        self.student = BaseModel(
            output_dim=output_dim,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            model_name=model_name,
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
        )

        self.teacher = BaseModel(
            output_dim=output_dim,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=True,
            model_name=model_name,
            patch_size=patch_size,
            # Here we do not apply drop path to the teacher, as per the original DINO paper
        )

        self.register_buffer("center", torch.zeros(1, output_dim))

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

        self.momentum_schedule = cosine_scheduler(
            momentum_teacher, 1, n_epochs, n_dataloader_steps
        )

        self.lr_schedule = cosine_scheduler(
            lr * (batch_size_per_gpu * world_size) / 256.0,
            min_lr,
            n_epochs,
            n_dataloader_steps,
            warmup_epochs=warmup_epochs,
        )

        self.wd_schedule = cosine_scheduler(
            weight_decay,
            weight_decay_end,
            n_epochs,
            n_dataloader_steps,
        )

        # ViT models do not use BatchNorm
        if model_name.startswith("resnet"):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # The teacher only takes the global views, which are always the first two elements
        teacher_output = self.teacher(x[:2])
        student_output = self.student(x)

        teacher_entropy, teacher_student_kl = self._compute_teacher_student_metrics(
            teacher_output, student_output
        )

        teacher_feats = teacher_output.detach()
        feat_var = torch.var(teacher_feats, dim=-1, unbiased=False)
        mean_feat_var = feat_var.mean()

        feat_l2 = torch.linalg.norm(teacher_feats, ord=2, dim=-1)
        mean_feat_l2 = feat_l2.mean()
        std_feat_l2 = feat_l2.std(unbiased=False)

        loss = self._calculate_loss(student_output, teacher_output)

        self.log("teacher_entropy", teacher_entropy, prog_bar=False, sync_dist=False)
        self.log(
            "teacher_student_kl", teacher_student_kl, prog_bar=False, sync_dist=False
        )
        self.log("teacher_feat_var_mean", mean_feat_var, prog_bar=False, sync_dist=False)
        self.log("teacher_feat_l2_mean", mean_feat_l2, prog_bar=False, sync_dist=False)
        self.log("teacher_feat_l2_std", std_feat_l2, prog_bar=False, sync_dist=False)
        self.log("train_loss", loss, prog_bar=True, sync_dist=False)

        return loss

    @torch.no_grad()
    def _compute_teacher_student_metrics(self, teacher_output, student_output):
        temp = self.teacher_temp_schedule[self.current_epoch]
        teacher_probs = F.softmax((teacher_output - self.center) / temp, dim=-1)
        student_scaled = student_output / self.student_temp
        student_log_probs = [
            F.log_softmax(chunk, dim=-1) for chunk in student_scaled.chunk(self.ncrops)
        ]

        eps = 1e-6
        teacher_entropy = (
            -(teacher_probs * torch.log(teacher_probs.clamp_min(eps))).sum(dim=-1)
        ).mean()

        teacher_chunks = teacher_probs.detach().chunk(2)
        kl_total = teacher_output.new_tensor(0.0)
        kl_terms = 0
        for iq, q in enumerate(teacher_chunks):
            log_q = torch.log(q.clamp_min(eps))
            for v, student_log_prob in enumerate(student_log_probs):
                if v == iq:
                    continue
                kl = torch.sum(q * (log_q - student_log_prob), dim=-1)
                kl_total += kl.mean()
                kl_terms += 1
        teacher_student_kl = kl_total / kl_terms if kl_terms > 0 else kl_total

        return teacher_entropy, teacher_student_kl

    @torch.no_grad()
    def _ema_update(self, global_step):
        m = self.momentum_schedule[global_step]
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _calculate_loss(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[self.current_epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        global_step = epoch * self.n_dataloader_steps + batch_idx
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[global_step]

        # normal optimizer step
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()

        self._ema_update(global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # Some parameters, like biases, were not regularized in the original DINO code
            get_params_groups(self.student),
        )
        return optimizer


class BaseModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        use_bn_in_head: bool,
        norm_last_layer: bool,
        model_name: str = "resnet18",
        patch_size: int | None = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
        elif model_name == "vit_tiny":
            self.backbone = vit_tiny(
                patch_size=patch_size, drop_path_rate=drop_path_rate
            )
        elif model_name == "vit_small":
            self.backbone = vit_small(
                patch_size=patch_size, drop_path_rate=drop_path_rate
            )
        elif model_name == "vit_base":
            self.backbone = vit_base(
                patch_size=patch_size, drop_path_rate=drop_path_rate
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if model_name.startswith("resnet"):
            embed_dim = self.backbone.fc.weight.shape[1]
        else:
            embed_dim = self.backbone.embed_dim

        self.backbone.fc = nn.Identity()
        # Some models use 'head' instead of 'fc' for the final layer
        # This should work if the model doesn't have a 'head' attribute
        self.backbone.head = nn.Identity()

        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=output_dim,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )

    def forward(self, x):
        # Here we are grouping inputs of different sizes to speed up the training.
        # We assume the models can handle images of different sizes, but since it's more
        # efficient to process images of the same size together, we group them.
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            output = torch.cat((output, _out))
            start_idx = end_idx
        return self.head(output)


# Adapted from: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn_in_head=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn_in_head:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn_in_head:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
