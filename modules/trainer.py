import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, DDPMPipeline, DDIMScheduler
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
import tarfile
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import io
from utils.telegram_notifier import notifier
from pathlib import Path
from scipy import ndimage
import pyvista as pv


PYVISTA_AVAILABLE = True  # hardcoded availability


class Trainer:
    def __init__(self, model, train_loader, val_loader, args, dataset, use_2d_unet, disable_telegram):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.dataset = dataset
        self.use_2d_unet = use_2d_unet
        self.disable_telegram = disable_telegram
        self.device = args.device

        # cache some samples from the dataset for generation context later
        self.cached_context_samples = None
        if hasattr(model.config, "inpainting_mode") and model.config.inpainting_mode:
            try:
                for batch in self.train_loader:
                    self.cached_context_samples = batch[:4].clone().detach()
                    break
            except Exception as e:
                print(f"Warning: Could not cache context samples: {e}")

        self.setup_training()

    def setup_training(self):
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.args.timesteps, beta_schedule=self.args.scheduler
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.diffusion_lr, weight_decay=1e-5)

        warmup_epochs = 5
        scheduler1 = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler2 = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs - warmup_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

        if self.args.accelerator:
            mixed_precision = "fp16" if self.args.mixed_precision else None
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            # prepare the model, optimizer, train_loader, val_loader, scheduler for distributed training
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
            self.device = self.accelerator.device
            if self.accelerator.is_main_process:
                print(f"Using Accelerator on device: {self.device}")
                print(f"Number of processes: {self.accelerator.num_processes}")
                print(f"Mixed precision: {self.accelerator.mixed_precision}")

                # initialize the chosen logger only on the main process
                if self.args.logger == "wandb":
                    try:
                        args_dict = vars(self.args).copy()
                        for key, value in args_dict.items():
                            if isinstance(value, Path):
                                args_dict[key] = str(value)
                        wandb.init(
                            project=self.args.project_name,
                            name=self.args.run_name,
                            config=args_dict,
                            mode=self.args.wandb_mode,
                            dir=str(self.args.output_dir),
                        )
                        print(
                            f"WandB initialized successfully for run: {self.args.run_name} (Mode: {self.args.wandb_mode}) -> Logs in {str(self.args.output_dir)}/wandb"
                        )
                    except Exception as e:
                        print(f"WandB initialization failed: {e}")
                        self.args.logger = "none"  # Fallback if init fails
                elif self.args.logger == "tensorboard":
                    try:
                        log_dir = os.path.join(self.args.output_dir, "tensorboard_logs")
                        self.summary_writer = SummaryWriter(log_dir=log_dir)
                        print(f"TensorBoard initialized. Logs will be saved to: {log_dir}")
                    except Exception as e:
                        print(f"TensorBoard initialization failed: {e}")
                        self.args.logger = "none"
                else:
                    print("Logging disabled.")
        else:
            self.model.to(self.device)
            # not using accelerator, using device: self.device which defaults to "cuda"
            print(f"Not using Accelerator. Using device: {self.device}")
            if self.args.logger == "wandb":
                try:
                    args_dict = vars(self.args).copy()
                    for key, value in args_dict.items():
                        if isinstance(value, Path):
                            args_dict[key] = str(value)
                    wandb.init(
                        project=self.args.project_name,
                        name=self.args.run_name,
                        config=args_dict,
                        mode=self.args.wandb_mode,
                        dir=str(self.args.output_dir),
                    )
                    print(
                        f"WandB initialized successfully for run: {self.args.run_name} (Mode: {self.args.wandb_mode}) -> Logs in {str(self.args.output_dir)}/wandb"
                    )
                except Exception as e:
                    print(f"WandB initialization failed: {e}")
                    self.args.logger = "none"
            elif self.args.logger == "tensorboard":
                try:
                    log_dir = os.path.join(self.args.output_dir, "tensorboard_logs")
                    self.summary_writer = SummaryWriter(log_dir=log_dir)
                    print(f"TensorBoard initialized. Logs will be saved to: {log_dir}")
                except Exception as e:
                    print(f"TensorBoard initialization failed: {e}")
                    self.args.logger = "none"
            else:
                print("Logging disabled.")

        # need a cleaner way to do this
        if self.args.logger != "tensorboard":
            self.summary_writer = None
        if self.args.logger != "wandb":
            pass

        self._logged_weighted_loss_info = False  # weighted loss logging flag

    def train(self):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        epochs_pbar = tqdm(
            range(self.args.epochs), desc="Training Epochs", disable=not self.accelerator.is_main_process
        )

        # wandb offline sync trigger hook
        trigger_sync = None
        if self.args.logger == "wandb" and self.args.wandb_mode == "offline":
            try:
                trigger_sync = TriggerWandbSyncHook()
                print("Initialized WandB offline sync trigger.")
            except Exception as e:
                print(f"Failed to initialize WandB sync hook: {e}")

        if self.accelerator.is_main_process and not self.disable_telegram:
            model_mode = "2D (Depth as Channels)" if self.use_2d_unet else "3D"
            notifier.send_message(
                f"Training Started ({model_mode})\nTotal epochs: {self.args.epochs}\nBatch size: {self.args.batch_size} (global)\nNum GPUs: {self.accelerator.num_processes if self.args.accelerator else 1}\nDevice: {self.device}"
            )

        for epoch in epochs_pbar:
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            self.accelerator.print(
                f"Epoch {epoch + 1}/{self.args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            if self.accelerator.is_main_process:
                epochs_pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)

            if self.accelerator.is_main_process:
                log_data = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                if self.args.logger == "wandb":
                    try:
                        wandb.log({**log_data, "epoch": epoch})
                    except Exception as e:
                        print(f"WandB scalar logging failed: {e}")
                elif self.args.logger == "tensorboard" and self.summary_writer:
                    try:
                        self.summary_writer.add_scalar("train_loss", train_loss, epoch)
                        self.summary_writer.add_scalar("val_loss", val_loss, epoch)
                        self.summary_writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
                    except Exception as e:
                        print(f"TensorBoard scalar logging failed: {e}")

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    self.log_reconstructions(epoch)
                    self.generate_sample_images(epoch)
                    if not self.disable_telegram:
                        self.send_telegram_loss_update(epoch, train_loss, val_loss)
                        self.send_telegram_image_comparison(epoch)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                save_model_flag = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_model_flag = True
                    if not self.disable_telegram:
                        self.send_telegram_loss_update(epoch, train_loss, val_loss, best=True)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.args.patience:
                        self.accelerator.print("Early stopping triggered.")
                        if not self.disable_telegram:
                            notifier.send_message(f"Early stopping triggered after {epochs_no_improve}.")
                        break

                if save_model_flag:
                    self.save_model()
                elif (epoch + 1) % 5 == 0:
                    self.save_model()

            # trigger wandb offline sync
            if self.accelerator.is_main_process and trigger_sync:
                try:
                    trigger_sync()
                except Exception as e:
                    print(f"WandB sync trigger failed: {e}")

            self.scheduler.step()

        if self.accelerator.is_main_process:
            if self.args.logger == "wandb":
                try:
                    wandb.finish()
                    print("WandB run finished.")
                except Exception as e:
                    print(f"WandB finish failed: {e}")
            elif self.args.logger == "tensorboard" and self.summary_writer:
                try:
                    self.summary_writer.close()
                    print("TensorBoard writer closed.")
                except Exception as e:
                    print(f"TensorBoard close failed: {e}")

            if not self.disable_telegram:
                notifier.send_message(f"✅ Training Completed\nBest validation loss: {best_val_loss:.4f}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} Training",
            disable=not self.accelerator.is_main_process,
        )
        batch_count = 0
        for batch in progress_bar:
            clean_images = batch
            if self.accelerator.is_main_process and batch_count == 0 and epoch == 0:
                # debug print to see if the voxel grids are in the correct -1 to 1 range
                print(
                    f"[Debug Epoch {epoch + 1}] Clean images min/max: {clean_images.min().item():.2f}/{clean_images.max().item():.2f}"
                )
            loss = self.train_step(clean_images, epoch)

            avg_loss = self.accelerator.gather(loss).mean().item()
            total_loss += avg_loss * clean_images.size(0)

            if self.accelerator.is_main_process:
                progress_bar.set_postfix(loss=avg_loss)

            batch_count += 1

        num_samples_in_loader = len(self.train_loader.dataset)
        avg_epoch_loss = total_loss / num_samples_in_loader

        return avg_epoch_loss

    def train_step(self, clean_images, epoch):
        """Performs a single training step, including noise addition, model prediction, and loss calculation."""

        # initialize mask variable for potential weighted loss
        mask_for_loss_weighting = None
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # batch size
        batch_size = clean_images.shape[0]

        # randn_noise sample
        noise = torch.randn_like(clean_images)

        # sample a random timestep for each image according to the chosen noise scheduler
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=clean_images.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # standard or inpainting mode
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model.config, "inpainting_mode") and unwrapped_model.config.inpainting_mode:
            # Inpainting Mode
            C = unwrapped_model.config.in_channels
            # generate random mask (e.g., block mask, free-form mask)
            mask_generator = MaskGenerator3D(args=self.args)
            mask = mask_generator(clean_images.shape).to(clean_images.device)  # shape (B, 1, D, H, W)
            mask_for_loss_weighting = mask  # store for potential weighted loss

            # create masked images (set masked areas to 0 or -1, depends on training)
            masked_images = clean_images * (1.0 - mask) - mask  # Set masked area to -1
            # TODO: test with -1 vs 0

            # concatenate: [noisy_latents, mask, masked_images]
            model_input = torch.cat([noisy_images, mask, masked_images], dim=1)  # shape (B, C*2+1, D, H, W)

        elif self.use_2d_unet:
            model_input = noisy_images.squeeze(1)  # Remove C=1 dim -> (B, D, H, W)
        else:
            model_input = noisy_images  # (B, C, D, H, W)

        # predict the noise residual
        noise_pred_tuple = self.model(model_input, timesteps, return_dict=False)
        noise_pred = noise_pred_tuple[0]

        if self.use_2d_unet:
            target_noise = noise.squeeze(1)  # (B, D, H, W) # the depth is used as the channel dimension
        else:
            target_noise = noise

        # calculate the loss
        # loss = F.mse_loss(noise_pred, target_noise) # standard mse loss

        use_weighted_loss_flag = getattr(self.args, "use_weighted_loss", False)
        inpainting_active_for_loss = (
            hasattr(unwrapped_model.config, "inpainting_mode")
            and unwrapped_model.config.inpainting_mode
            and mask_for_loss_weighting is not None
        )

        if use_weighted_loss_flag:
            if self.accelerator.is_main_process and not self._logged_weighted_loss_info:
                base_msg = f"Tariner with 'use_weighted_loss' is True. "
                weight_val_msg = f"Configured masked_loss_weight: {getattr(self.args, 'masked_loss_weight', 2.0)}. "
                print(base_msg + weight_val_msg + "will apply if inpainting conditions met and shapes match.")
                self._logged_weighted_loss_info = True

            if inpainting_active_for_loss:
                squared_error = (noise_pred - target_noise) ** 2
                weights = torch.ones_like(squared_error, device=squared_error.device)
                loss_mask_to_adapt = mask_for_loss_weighting.to(squared_error.device)

                if self.use_2d_unet:
                    adapted_loss_weights_mask = loss_mask_to_adapt.squeeze(1)
                else:
                    num_channels_target = target_noise.shape[1]
                    adapted_loss_weights_mask = loss_mask_to_adapt.repeat(1, num_channels_target, 1, 1, 1)

                if adapted_loss_weights_mask.shape == weights.shape:
                    masked_weight = getattr(self.args, "masked_loss_weight", 2.0)
                    weights = torch.where(adapted_loss_weights_mask > 0.5, masked_weight, 1.0)
                    loss = (weights * squared_error).mean()
                else:
                    # fallback if shapes mismatch for weighting, use standard mse loss
                    loss = F.mse_loss(noise_pred, target_noise)
            else:
                # only mse for non-inpainting cases
                loss = F.mse_loss(noise_pred, target_noise)
        else:
            # use_weighted_loss_flag is False, use standard MSE
            loss = F.mse_loss(noise_pred, target_noise)

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.detach()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch + 1} Validation",
                disable=not self.accelerator.is_main_process,
            )
            for batch in val_pbar:
                clean_images = batch
                loss = self.validation_step(clean_images, epoch)

                avg_loss = self.accelerator.gather(loss).mean().item()
                total_loss += avg_loss * clean_images.size(0)

                if self.accelerator.is_main_process:
                    val_pbar.set_postfix(loss=avg_loss)

        num_samples_in_loader = len(self.val_loader.dataset)
        avg_val_loss = total_loss / num_samples_in_loader

        return avg_val_loss

    def validation_step(self, clean_images, epoch):
        """Follows the same logic as train_step"""
        mask_for_loss_weighting = None
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        clean_images = clean_images.float().to(self.device)
        batch_size = clean_images.shape[0]
        B, C_orig, D, H, W = clean_images.shape

        noise = torch.randn_like(clean_images)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=clean_images.device
        ).long()

        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model.config, "inpainting_mode") and unwrapped_model.config.inpainting_mode:
            C = unwrapped_model.config.in_channels
            mask_generator = MaskGenerator3D(args=self.args)
            mask = mask_generator(clean_images.shape).to(clean_images.device)
            mask_for_loss_weighting = mask
            masked_images = clean_images * (1.0 - mask) - mask
            model_input = torch.cat([noisy_images, mask, masked_images], dim=1)

        elif self.use_2d_unet:
            model_input = noisy_images.squeeze(1)
        else:
            model_input = noisy_images  # 3d model

        # predict noise
        model_output_tuple = self.model(model_input, timesteps, return_dict=False)
        noise_pred_raw = model_output_tuple[0]

        if self.use_2d_unet:
            target_noise = noise.squeeze(1)
            noise_pred = noise_pred_raw
        else:
            target_noise = noise
            noise_pred = noise_pred_raw

        use_weighted_loss_flag = getattr(self.args, "use_weighted_loss", False)
        inpainting_active_for_loss = (
            hasattr(unwrapped_model.config, "inpainting_mode")
            and unwrapped_model.config.inpainting_mode
            and mask_for_loss_weighting is not None
        )

        if use_weighted_loss_flag:
            if inpainting_active_for_loss:
                squared_error = (noise_pred - target_noise) ** 2
                weights = torch.ones_like(squared_error, device=squared_error.device)
                loss_mask_to_adapt = mask_for_loss_weighting.to(squared_error.device)

                if self.use_2d_unet:
                    adapted_loss_weights_mask = loss_mask_to_adapt.squeeze(1)
                else:
                    num_channels_target = target_noise.shape[1]
                    adapted_loss_weights_mask = loss_mask_to_adapt.repeat(1, num_channels_target, 1, 1, 1)

                if adapted_loss_weights_mask.shape == weights.shape:
                    masked_weight = getattr(self.args, "masked_loss_weight", 2.0)
                    weights = torch.where(adapted_loss_weights_mask > 0.5, masked_weight, 1.0)
                    loss = (weights * squared_error).mean()
                else:
                    loss = F.mse_loss(noise_pred, target_noise)  # Fallback
            else:
                loss = F.mse_loss(noise_pred, target_noise)
        else:
            loss = F.mse_loss(noise_pred, target_noise)

        if self.accelerator.is_main_process and (epoch % 5 == 0 or epoch == 0):
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
            beta_prod_t = 1 - alpha_prod_t
            noise_pred_for_recon = noise_pred
            if self.use_2d_unet:
                noise_pred_for_recon = noise_pred.unsqueeze(1)

            pred_original_sample = (noisy_images - beta_prod_t.sqrt() * noise_pred_for_recon) / alpha_prod_t.sqrt()
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
            self.recon_images = pred_original_sample.detach()
            self.clean_images = clean_images.detach()

        return loss.detach()

    def log_reconstructions(self, epoch):
        if (
            not self.accelerator.is_main_process
            or not hasattr(self, "clean_images")
            or not hasattr(self, "recon_images")
        ):
            return

        try:
            # Get samples (already detached, 5D: B, 1, D, H, W)
            clean_sample_5d = self.clean_images[:4].cpu()
            recon_sample_5d = self.recon_images[:4].cpu()

            #  Simplified Scaling
            processed_clean = []
            processed_recon = []

            for i in range(clean_sample_5d.shape[0]):
                clean_vol_norm = clean_sample_5d[i, 0].numpy()  # (D, H, W)
                recon_vol_norm = recon_sample_5d[i, 0].numpy()  # (D, H, W)

                # Always scale from [-1, 1] to [0, 1] for visualization
                clean_vol_01 = (clean_vol_norm + 1.0) / 2.0
                recon_vol_01 = (recon_vol_norm + 1.0) / 2.0

                # Ensure values are clipped to [0, 1] for visualization
                clean_vol_01 = np.clip(clean_vol_01, 0.0, 1.0)
                recon_vol_01 = np.clip(recon_vol_01, 0.0, 1.0)

                processed_clean.append(torch.tensor(clean_vol_01))
                processed_recon.append(torch.tensor(recon_vol_01))

            # Stack back into (B, D, H, W)
            processed_clean_tensor = torch.stack(processed_clean).unsqueeze(1)  # Add channel dim
            processed_recon_tensor = torch.stack(processed_recon).unsqueeze(1)
            #

            # Select middle slice along depth (D) axis for visualization
            depth_slice_idx = processed_clean_tensor.shape[2] // 2
            clean_slices = processed_clean_tensor[:, :, depth_slice_idx, :, :]  # (B, 1, H, W)
            recon_slices = processed_recon_tensor[:, :, depth_slice_idx, :, :]

            # Create a grid (simple concatenation for now)
            comparison = torch.cat([clean_slices, recon_slices], dim=0)  # (2*B, 1, H, W)

            #  Conditional Image Logging
            if self.args.logger == "wandb":
                try:
                    wandb.log(
                        {
                            "clean_vs_recon_slices": wandb.Image(
                                comparison, caption=f"Epoch {epoch + 1}: Recon vs Clean (Middle Slices)"
                            ),
                            "epoch": epoch,
                        }
                    )
                except Exception as e:
                    print(f"WandB image logging failed: {e}")
            elif self.args.logger == "tensorboard" and self.summary_writer:
                try:
                    self.summary_writer.add_images("clean_vs_recon_slices", comparison, epoch, dataformats="NCHW")
                except Exception as e:
                    print(f"TensorBoard image logging failed: {e}")

        except Exception as e:
            self.accelerator.print(f"Error logging reconstructions: {e}")

    def save_model(self, checkpoint=False, epoch=None):
        if not self.accelerator.is_main_process:
            return

        save_dir = os.path.join(self.args.output_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        if checkpoint and epoch is not None:
            chkpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}")
            self.accelerator.save_state(chkpt_path)
            self.accelerator.print(f"Checkpoint saved to {chkpt_path}")
        else:
            # For the sake of compatibility with the diffusers pipeline, we store the model directly as the DDPMPipeline
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            pipeline = DDPMPipeline(unet=unwrapped_model, scheduler=self.noise_scheduler)
            pipeline.save_pretrained(self.args.output_dir)
            self.accelerator.print(f"Model saved to {self.args.output_dir}")

            args_dict = vars(self.args).copy()
            for key, value in args_dict.items():
                if isinstance(value, Path):
                    args_dict[key] = str(value)

            with open(os.path.join(str(self.args.output_dir), "training_args.json"), "w") as f:
                json.dump(args_dict, f, indent=4)

    def send_telegram_loss_update(self, epoch, train_loss, val_loss, best=False):
        if not self.accelerator.is_main_process or self.disable_telegram:
            return
        status = " Best Model Saved" if best else "Epoch Update"
        message = f"{status} - Epoch {epoch + 1}\nTrain Loss: {train_loss:.4f}\nVal Loss: {val_loss:.4f}"
        try:
            notifier.send_message(message)
        except Exception as e:
            self.accelerator.print(f"Failed to send Telegram message: {e}")

    def send_telegram_image_comparison(self, epoch):
        if (
            not self.accelerator.is_main_process
            or self.disable_telegram
            or not hasattr(self, "clean_images")
            or not hasattr(self, "recon_images")
        ):
            return

        try:
            # Take one sample (5D: 1, 1, D, H, W)
            clean_sample_5d = self.clean_images[0].cpu()
            recon_sample_5d = self.recon_images[0].cpu()

            #  Simplified Scaling (Assume input/output is [-1, 1])
            clean_vol_norm = clean_sample_5d[0].numpy()  # (D, H, W)
            recon_vol_norm = recon_sample_5d[0].numpy()  # (D, H, W)

            # Always scale from [-1, 1] to [0, 1] for visualization
            clean_vol_01 = (clean_vol_norm + 1.0) / 2.0
            recon_vol_01 = (recon_vol_norm + 1.0) / 2.0

            clean_vol_01 = np.clip(clean_vol_01, 0.0, 1.0)
            recon_vol_01 = np.clip(recon_vol_01, 0.0, 1.0)
            #

            # Select middle slices (Depth, Height, Width)
            d_slice_idx = clean_vol_01.shape[0] // 2
            h_slice_idx = clean_vol_01.shape[1] // 2
            w_slice_idx = clean_vol_01.shape[2] // 2

            clean_slice_d = clean_vol_01[d_slice_idx, :, :]
            recon_slice_d = recon_vol_01[d_slice_idx, :, :]
            clean_slice_h = clean_vol_01[:, h_slice_idx, :]
            recon_slice_h = recon_vol_01[:, h_slice_idx, :]
            clean_slice_w = clean_vol_01[:, :, w_slice_idx]
            recon_slice_w = recon_vol_01[:, :, w_slice_idx]

            #  Plotting (remains the same, uses *_vol_01 slices)
            fig, axs = plt.subplots(3, 2, figsize=(8, 10))
            fig.suptitle(f"Epoch {epoch + 1}: Clean vs. Recon (Middle Slices)")

            axs[0, 0].imshow(clean_slice_d, cmap="gray", vmin=0, vmax=1)
            axs[0, 0].set_title("Clean (Depth Slice)")
            axs[0, 0].axis("off")
            axs[0, 1].imshow(recon_slice_d, cmap="gray", vmin=0, vmax=1)
            axs[0, 1].set_title("Recon (Depth Slice)")
            axs[0, 1].axis("off")

            axs[1, 0].imshow(clean_slice_h, cmap="gray", vmin=0, vmax=1)
            axs[1, 0].set_title("Clean (Height Slice)")
            axs[1, 0].axis("off")
            axs[1, 1].imshow(recon_slice_h, cmap="gray", vmin=0, vmax=1)
            axs[1, 1].set_title("Recon (Height Slice)")
            axs[1, 1].axis("off")

            axs[2, 0].imshow(clean_slice_w, cmap="gray", vmin=0, vmax=1)
            axs[2, 0].set_title("Clean (Width Slice)")
            axs[2, 0].axis("off")
            axs[2, 1].imshow(recon_slice_w, cmap="gray", vmin=0, vmax=1)
            axs[2, 1].set_title("Recon (Width Slice)")
            axs[2, 1].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)

            buf.seek(0)

            notifier.send_image(buf.getvalue(), caption=f"Epoch {epoch + 1} Reconstruction Comparison")

            buf.close()

        except Exception as e:
            self.accelerator.print(f"Error during Telegram image comparison (sending skipped if disabled): {e}")

    def generate_sample_images(self, epoch):
        if not self.accelerator.is_main_process:
            return

        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            num_samples = 4
            num_inference_steps = self.args.timesteps

            # Determine shape from model config or dataset/args
            if self.use_2d_unet:
                # For 2D UNet, D is treated as channels. Sample size is (H, W).
                H, W = unwrapped_model.config.sample_size
                D = unwrapped_model.config.in_channels  # Depth is stored in in_channels
                C = 1  # The actual channel dim for the 5D tensor is 1
            else:
                # For 3D UNet, sample_size is (D, H, W).
                if isinstance(unwrapped_model.config.sample_size, int):
                    D = H = W = unwrapped_model.config.sample_size
                else:
                    # Assuming tuple (D, H, W)
                    D, H, W = unwrapped_model.config.sample_size
                C = unwrapped_model.config.in_channels  # Should be 1 for 3D UNet case

            image_shape = (num_samples, C, D, H, W)

            # For inpainting mode, use cached context samples
            context_samples = None
            is_inpainting_unet = (
                (not self.use_2d_unet)
                and hasattr(unwrapped_model.config, "inpainting_mode")
                and unwrapped_model.config.inpainting_mode
            )

            if is_inpainting_unet:
                # Use the cached samples from initialization (safe for distributed training)
                if self.cached_context_samples is not None:
                    context_samples = self.cached_context_samples.to(self.device)
                    # Make sure it has the right number of samples
                    if context_samples.shape[0] > num_samples:
                        context_samples = context_samples[:num_samples]
                    elif context_samples.shape[0] < num_samples:
                        # If we have fewer cached samples than needed, repeat the last one
                        repeats = num_samples - context_samples.shape[0]
                        context_samples = torch.cat(
                            [context_samples, context_samples[-1:].repeat(repeats, 1, 1, 1, 1)], dim=0
                        )

            # Initialize noise
            generator = torch.Generator(device=self.device).manual_seed(self.args.seed + epoch)
            latents = torch.randn(image_shape, generator=generator, device=self.device)

            #  Initialize Sampler based on args
            sampler_type = getattr(self.args, "sampler_type", "ddpm").lower()

            if sampler_type == "ddim":
                sampler = DDIMScheduler.from_config(self.noise_scheduler.config)
                tqdm_desc = "Generating Samples (DDIM)"
            else:  # Default to ddpm
                sampler = self.noise_scheduler
                tqdm_desc = "Generating Samples (DDPM)"

            sampler.set_timesteps(num_inference_steps, device=self.device)
            sampler_timesteps_array = sampler.timesteps
            #

            # Main sampling loop
            sampling_pbar = tqdm(
                enumerate(sampler_timesteps_array),
                total=num_inference_steps,
                desc=tqdm_desc,
                disable=not self.accelerator.is_main_process,
            )
            with torch.no_grad():
                for i, t in sampling_pbar:
                    t_input = t.repeat(num_samples)
                    latents_device = latents.to(self.device)

                    #  Adapt model input based on UNet type
                    if is_inpainting_unet:
                        # Create partial mask (some known, some unknown)
                        # Instead of all ones (all unknown), create a mask with some knowns

                        # Define a percentage of the volume to keep as context (e.g., 30%)
                        context_percentage = 0.5  # Increase to 50% for more structure

                        if context_samples is not None:
                            # Use real dataset samples as context
                            mask_generator = MaskGenerator3D(args=self.args)
                            # Generate mask - we want a consistent mask across the batch
                            temp_shape = latents_device.shape
                            mask = mask_generator([1, 1, D, H, W]).to(latents_device.device)
                            # Expand to batch size
                            mask = mask.repeat(num_samples, 1, 1, 1, 1)

                            # Save the mask and context for visualization (do this only for the first timestep)
                            if t == sampler_timesteps_array[0]:
                                self.saved_gen_mask = mask.clone().cpu()
                                self.saved_context = context_samples.clone().cpu()
                                print(
                                    f"Saved generation mask and context for visualization (shapes: {self.saved_gen_mask.shape}, {self.saved_context.shape})"
                                )

                            # Create masked_images with dataset sample values where mask is 0 (known)
                            # and current latent values where mask is 1 (unknown)
                            masked_images = context_samples * (1.0 - mask) + latents_device * mask
                        else:
                            # Fallback if no context samples: create partially random context
                            # Use edge mask
                            mask_generator = MaskGenerator3D(args=self.args)
                            mask = mask_generator([num_samples, 1, D, H, W]).to(latents_device.device)

                            # Generate some structure to use as context
                            masked_images = torch.zeros_like(latents_device)
                            for b in range(num_samples):
                                # Create simple geometric primitives for context
                                # Example: Add a sphere to provide context
                                center_d = torch.randint(D // 4, 3 * D // 4, (1,)).item()
                                center_h = torch.randint(H // 4, 3 * H // 4, (1,)).item()
                                center_w = torch.randint(W // 4, 3 * W // 4, (1,)).item()
                                radius = min(D, H, W) // 6

                                # Create coordinates
                                d_coords = torch.arange(D).unsqueeze(1).unsqueeze(1).repeat(1, H, W)
                                h_coords = torch.arange(H).unsqueeze(0).unsqueeze(2).repeat(D, 1, W)
                                w_coords = torch.arange(W).unsqueeze(0).unsqueeze(0).repeat(D, H, 1)

                                # Calculate distance from center
                                dist = torch.sqrt(
                                    (d_coords - center_d) ** 2
                                    + (h_coords - center_h) ** 2
                                    + (w_coords - center_w) ** 2
                                )

                                # Create a sphere
                                sphere = (dist < radius).float() * 2 - 1  # Scale to [-1, 1]

                                # Add the sphere to the context (only in known regions)
                                masked_images[b, 0] = sphere * (1.0 - mask[b, 0]) - mask[b, 0]

                        # Concatenate inputs for inpainting UNet
                        model_input = torch.cat([latents_device, mask, masked_images], dim=1)
                    elif self.use_2d_unet:
                        # Standard 2D UNet Mode (Depth as Channels)
                        model_input = latents_device.squeeze(1)  # Remove C=1 dim -> (B, D, H, W)
                    else:
                        # Standard 3D UNet Mode (Non-inpainting)
                        model_input = latents_device  # (B, C, D, H, W)
                    #

                    noise_pred_tuple = unwrapped_model(model_input, t_input, return_dict=False)
                    noise_pred_raw = noise_pred_tuple[0].to(self.device)  # Ensure output is on device

                    #  Adapt model output based on UNet type
                    if self.use_2d_unet:
                        # Output is (B, D, H, W), reshape back to (B, 1, D, H, W) for scheduler
                        noise_pred = noise_pred_raw.unsqueeze(1)  # Add C=1 dim
                    else:
                        # Output is (B, C, D, H, W) for 3D UNet
                        noise_pred = noise_pred_raw
                    #

                    # Use the chosen sampler's step function
                    if sampler_type == "ddim":
                        step_output = sampler.step(noise_pred, t, latents_device, eta=0.0)
                    else:  # DDPM
                        step_output = sampler.step(noise_pred, t, latents_device)
                    x_prev_denoised_candidate = step_output.prev_sample

                    #  RePaint-like step for inpainting guidance (works with DDPM/DDIM)
                    if is_inpainting_unet and context_samples is not None and hasattr(self, "saved_gen_mask"):
                        context_samples_device = context_samples.to(self.device)
                        inpainting_mask_device = self.saved_gen_mask.to(self.device)
                        if x_prev_denoised_candidate.shape[1] != inpainting_mask_device.shape[1]:
                            inpainting_mask_device = inpainting_mask_device.repeat(
                                1, x_prev_denoised_candidate.shape[1], 1, 1, 1
                            )

                        if i < num_inference_steps - 1:
                            prev_t_val_for_add_noise = sampler_timesteps_array[i + 1]
                            noise_for_gt_conditioning = torch.randn_like(context_samples_device, device=self.device)
                            prev_t_batch = prev_t_val_for_add_noise.repeat(num_samples).long()

                            # IMPORTANT: Always use the original DDPMScheduler (self.noise_scheduler) for adding noise
                            # as this matches the forward process during training.
                            x_prev_noised_known_gt = self.noise_scheduler.add_noise(
                                context_samples_device, noise_for_gt_conditioning, prev_t_batch
                            )

                            latents = x_prev_denoised_candidate * inpainting_mask_device + x_prev_noised_known_gt * (
                                1.0 - inpainting_mask_device
                            )
                        else:
                            # Last step: composite final x0 prediction for masked regions
                            latents = x_prev_denoised_candidate * inpainting_mask_device + context_samples_device * (
                                1.0 - inpainting_mask_device
                            )
                    else:
                        # Standard step if not inpainting or context/mask not available
                        latents = x_prev_denoised_candidate
                    #  End RePaint-like step

            # Convert to numpy
            images_5d_np = latents.cpu().float().numpy()
            #

            # Ensure C=1 for grayscale visualization/processing below
            if images_5d_np.shape[1] != 1:
                print(f"Warning: Generated samples have {images_5d_np.shape[1]} channels. Processing assumes 1.")

            #  Simplified Scaling (Assume input/output is [-1, 1])
            processed_volumes_01 = []  # Store processed volumes in [0, 1] range

            for i in range(images_5d_np.shape[0]):
                vol_norm = images_5d_np[i, 0]  # Take first channel -> (D, H, W)

                # Always scale from [-1, 1] to [0, 1] for saving/visualization
                vol_01 = (vol_norm + 1.0) / 2.0

                vol_01 = np.clip(vol_01, 0.0, 1.0)

                # Apply Gaussian smoothing to reduce noise before thresholding
                # Apply light smoothing with sigma=0.5
                vol_01_smooth = ndimage.gaussian_filter(vol_01, sigma=0.5)

                # Apply threshold
                vol_binary = (vol_01_smooth > 0.5).astype(np.int32)

                # Optional: Remove small isolated components (noise)
                # Label connected components
                labeled_array, num_features = ndimage.label(vol_binary)

                # Count voxels in each component
                component_sizes = np.bincount(labeled_array.ravel())

                # Set a minimum size threshold (e.g., components smaller than 1% of volume)
                min_size = int(0.01 * vol_binary.size)

                # Remove small components (keep component 0 which is background)
                too_small = np.zeros(component_sizes.shape, bool)
                too_small[1:] = component_sizes[1:] < min_size

                # Remove components that are too small
                vol_binary_clean = vol_binary.copy()
                for label in np.where(too_small)[0]:
                    vol_binary_clean[labeled_array == label] = 0

                # Convert back to float32
                vol_01_final = vol_binary_clean.astype(np.float32)
                processed_volumes_01.append(vol_01_final)
            #

            #  Save Generated Volumes as Compressed VTU Archive (Optional)
            if PYVISTA_AVAILABLE:
                vtu_dir = Path(self.args.output_dir) / "generated_vtu"  # Main dir for archives
                vtu_dir.mkdir(parents=True, exist_ok=True)
                temp_vtu_dir = vtu_dir / f"epoch_{epoch:04d}_temp"  # Temp dir for this epoch
                temp_vtu_dir.mkdir(exist_ok=True)

                saved_vtu_files = []
                for i, volume_np_0_1 in enumerate(processed_volumes_01):
                    try:
                        # Get shape D, H, W from the numpy volume
                        D, H, W = volume_np_0_1.shape
                        grid = pv.ImageData()
                        grid.dimensions = np.array([W, H, D]) + 1
                        grid.origin = (0, 0, 0)
                        grid.spacing = (1, 1, 1)
                        grid.cell_data["values"] = np.ascontiguousarray(volume_np_0_1).flatten(order="C")

                        # Define temporary save path with .vti extension
                        vti_filename = f"epoch_{epoch:04d}_sample_{i:02d}.vti"
                        temp_vti_save_path = temp_vtu_dir / vti_filename

                        # Save the grid temporarily
                        grid.save(str(temp_vti_save_path), binary=True)
                        saved_vtu_files.append(str(temp_vti_save_path))

                    except Exception as e:
                        # Update error message to mention VTI
                        self.accelerator.print(f"Error saving temporary VTI file for epoch {epoch} sample {i}: {e}")

                # Compress the temporary VTI files into a single archive
                if saved_vtu_files:  # Proceed only if some files were saved
                    # Keep archive name as .tar.gz, content type is implicit
                    archive_filename = vtu_dir / f"epoch_{epoch:04d}.tar.gz"
                    try:
                        print(f"Compressing {len(saved_vtu_files)} VTI files to {archive_filename}...")
                        with tarfile.open(archive_filename, "w:gz") as tar:
                            for vti_file_path in saved_vtu_files:
                                # Add file to archive using just the filename as arcname
                                tar.add(vti_file_path, arcname=os.path.basename(vti_file_path))
                        print(f"Compression complete for epoch {epoch}.")
                    except Exception as e:
                        self.accelerator.print(f"Error creating VTI archive for epoch {epoch}: {e}")
                    finally:
                        # Clean up the temporary directory regardless of compression success
                        try:
                            shutil.rmtree(temp_vtu_dir)
                            # print(f"Removed temporary directory: {temp_vtu_dir}") # Optional debug
                        except Exception as e:
                            self.accelerator.print(f"Error removing temporary VTU directory {temp_vtu_dir}: {e}")
                else:
                    # If no files were saved, still clean up the temp dir
                    try:
                        shutil.rmtree(temp_vtu_dir)
                    except Exception as e:
                        self.accelerator.print(f"Error removing empty temporary VTU directory {temp_vtu_dir}: {e}")
            #        Visualization using NumPy arrays
            # Select middle depth slice for visualization
            depth_slice_idx = processed_volumes_01[0].shape[0] // 2
            image_slices_np = [vol[depth_slice_idx, :, :] for vol in processed_volumes_01]

            # Convert to tensors for TensorBoard logging
            slices_tensor_list = [torch.tensor(s).unsqueeze(0) for s in image_slices_np]  # Add channel dim
            slices_grid = torch.stack(slices_tensor_list)  # (B, 1, H, W)

            #  Add visualization of masked inputs (context) for inpainting mode
            if is_inpainting_unet and context_samples is not None:
                print(
                    f"Preparing inpainting visualization, has saved data: {hasattr(self, 'saved_gen_mask')=}, {hasattr(self, 'saved_context')=}"
                )
                masked_input_slices = []
                raw_generated_slices = []
                final_generated_slices = []  # Reuse existing processed_volumes_01

                # Use the saved mask and context from the sampling loop
                if hasattr(self, "saved_gen_mask") and hasattr(self, "saved_context"):
                    with torch.no_grad():
                        try:
                            context_tensor = self.saved_context.cpu()
                            gen_mask = self.saved_gen_mask.cpu()
                            masked_viz = context_tensor * (1.0 - gen_mask)
                            masked_viz_np = masked_viz.numpy()  # Shape NCDHW
                            gen_mask_np = gen_mask.numpy()

                            # Also get the raw generated volumes
                            # images_5d_np is the raw output scaled to [-1, 1]
                            raw_volumes_01_np = (images_5d_np + 1.0) / 2.0  # Scale to [0, 1]
                            raw_volumes_01_np = np.clip(raw_volumes_01_np, 0.0, 1.0)

                            # Calculate mask content for each sample and slice
                            num_samples = gen_mask_np.shape[0]
                            D = gen_mask_np.shape[2]
                            H = gen_mask_np.shape[3]
                            W = gen_mask_np.shape[4]

                            # Sum of mask content for each slice across all samples
                            all_mask_sums = []
                            sample_mask_sums = []
                            for sample_idx in range(num_samples):
                                # Sum over H, W for each D slice in this sample
                                mask_sum_per_slice = np.sum(gen_mask_np[sample_idx, 0], axis=(1, 2))
                                sample_mask_sums.append(mask_sum_per_slice)
                                all_mask_sums.append(mask_sum_per_slice)

                            # Find the best slice across all samples (most mask content)
                            all_mask_sums = np.concatenate(all_mask_sums)
                            best_slice_idx = np.argmax(all_mask_sums) % D
                            print(f"Selected slice {best_slice_idx} with most mask content across all samples")

                            for sample_idx in range(num_samples):
                                # Check if this slice has meaningful mask content for this sample
                                mask_content = sample_mask_sums[sample_idx][best_slice_idx]
                                mask_threshold = 0.0001 * H * W

                                if mask_content > mask_threshold:
                                    # Use the globally selected slice
                                    slice_idx = best_slice_idx
                                    print(
                                        f"Sample {sample_idx}: Using common slice {slice_idx} with mask content {mask_content:.1f} px"
                                    )
                                else:
                                    sample_best_slice = np.argmax(sample_mask_sums[sample_idx])
                                    slice_idx = sample_best_slice
                                    print(
                                        f"Sample {sample_idx}: No content in common slice, using sample-specific slice {slice_idx}"
                                    )

                                mask_slice = gen_mask_np[sample_idx, 0, slice_idx]

                                # Scale the context to [0,1] range for visualization
                                context_slice_01 = (masked_viz_np[sample_idx, 0, slice_idx] + 1.0) / 2.0
                                context_slice_01 = np.clip(context_slice_01, 0.0, 1.0)

                                colored_context = np.zeros((H, W, 3), dtype=np.float32)
                                colored_context[:, :, 0] = context_slice_01  # Context in red
                                colored_context[:, :, 1] = context_slice_01  # Context in green
                                colored_context[:, :, 2] = context_slice_01  # Context in blue

                                # Apply mask as white overlay
                                mask_color = np.zeros((H, W, 3), dtype=np.float32)
                                mask_color[:, :, 0] = 0.3  # Red component
                                mask_color[:, :, 1] = 0.9  # Green component (stronger)
                                mask_color[:, :, 2] = 0.3  # Blue component

                                # Blend mask with context - more white for better visibility
                                for c in range(3):
                                    colored_context[:, :, c] = (1 - mask_slice) * colored_context[
                                        :, :, c
                                    ] + mask_slice * mask_color[:, :, c]

                                masked_input_slices.append(colored_context)

                                # 2. Raw Generated Slice
                                raw_gen_slice = raw_volumes_01_np[sample_idx, 0, slice_idx]
                                raw_generated_slices.append(raw_gen_slice)

                                # 3. Final Generated Slice
                                final_gen_slice = processed_volumes_01[sample_idx][slice_idx]
                                final_generated_slices.append(final_gen_slice)

                            # Create combined visualization with all three views side by side
                            if self.args.logger == "wandb" and wandb:
                                for i in range(len(masked_input_slices)):
                                    # Create a side-by-side visualization
                                    combined_height = H
                                    combined_width = W * 3 + 20
                                    combined_viz = np.ones((combined_height, combined_width, 3), dtype=np.float32)

                                    combined_viz[:, :W, :] = masked_input_slices[i]

                                    raw_slice = raw_generated_slices[i]
                                    combined_viz[:, W + 10 : 2 * W + 10, 0] = raw_slice
                                    combined_viz[:, W + 10 : 2 * W + 10, 1] = raw_slice
                                    combined_viz[:, W + 10 : 2 * W + 10, 2] = raw_slice

                                    final_slice = final_generated_slices[i]
                                    combined_viz[:, 2 * W + 20 : 3 * W + 20, 0] = final_slice
                                    combined_viz[:, 2 * W + 20 : 3 * W + 20, 1] = final_slice
                                    combined_viz[:, 2 * W + 20 : 3 * W + 20, 2] = final_slice

                                    # Add labels
                                    wandb.log(
                                        {
                                            f"inpainting_combined_{i}": wandb.Image(
                                                combined_viz,
                                                caption=f"Sample {i}: Left=Masked Input, Middle=Raw Output, Right=Processed",
                                            )
                                        },
                                        commit=False,
                                    )
                        except Exception as e:
                            print(f"Error creating mask projection visualization: {e}")
                            masked_input_slices = None
                else:
                    print("Warning: No saved masks/context for visualization")
                    masked_input_slices = None

            #  Conditional Image Logging
            # Also log the max projections of the generated samples
            generated_projections_np = []
            if processed_volumes_01:
                for vol in processed_volumes_01:
                    proj = np.max(vol, axis=0)
                    generated_projections_np.append(proj)

            if self.args.logger == "wandb":
                try:
                    # Log generated projections as grayscale images
                    if generated_projections_np:
                        wandb.log(
                            {
                                f"generated_proj_{i}": wandb.Image(
                                    proj, caption=f"Epoch {epoch + 1}: Generated Projection {i}"
                                )
                                for i, proj in enumerate(generated_projections_np)
                            },
                            commit=False,
                        )

                    wandb.log({"epoch": epoch})

                except Exception as e:
                    print(f"WandB image projection logging failed: {str(e)}")
            elif self.args.logger == "tensorboard" and self.summary_writer:
                try:
                    # Log generated projections (needs channel dim: NCHW)
                    if generated_projections_np:
                        gen_proj_tensor = torch.from_numpy(np.array(generated_projections_np)).unsqueeze(
                            1
                        )  # Add C dim
                        self.summary_writer.add_images(
                            "generated_projections", gen_proj_tensor, epoch, dataformats="NCHW"
                        )

                    # Log masked inputs if available
                    if is_inpainting_unet and masked_input_slices:
                        masked_proj_array = np.array(masked_input_slices)  # Shape: [N, H, W, 3]
                        masked_proj_tensor = torch.from_numpy(masked_proj_array).permute(0, 3, 1, 2)  # NHWC -> NCHW
                        self.summary_writer.add_images(
                            "masked_input_projections", masked_proj_tensor, epoch, dataformats="NCHW"
                        )
                except Exception as e:
                    print(f"TensorBoard image projection logging failed: {str(e)}")

            # Create plot (for Telegram) - keep this regardless of logger
            # For inpainting, show both input context and results side by side using projections
            try:
                if is_inpainting_unet and masked_input_slices:
                    print(
                        f"Matplotlib plotting projections: num_samples={num_samples}, masked_input_slices length={len(masked_input_slices)}"
                    )

                    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
                    fig.suptitle(f"Epoch {epoch + 1} Inpainting: Masked Proj (top) vs Generated Proj (bottom)")

                    if num_samples == 1:
                        axs = axs.reshape(2, 1)

                    for i in range(num_samples):
                        if i < len(masked_input_slices) and i < len(generated_projections_np):
                            # Top row: Masked input projections
                            axs[0, i].imshow(masked_input_slices[i])
                            axs[0, i].set_title("Masked Input Proj")
                            axs[0, i].axis("off")

                            # Bottom row: Generated result projections
                            axs[1, i].imshow(generated_projections_np[i], cmap="gray", vmin=0, vmax=1)
                            axs[1, i].set_title("Generated Proj")
                            axs[1, i].axis("off")
                        else:
                            print(f"Warning: Index {i} out of range for projections")
                else:
                    # Standard visualization
                    if generated_projections_np:
                        fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
                        fig.suptitle(f"Epoch {epoch + 1} Generated Projections")

                        if num_samples == 1:
                            axs = np.array([axs])

                        for i in range(num_samples):
                            if i < len(generated_projections_np):
                                proj_np = generated_projections_np[i]
                                axs[i].imshow(proj_np, cmap="gray", vmin=0, vmax=1)
                                axs[i].axis("off")
                            else:
                                print(f"Warning: Index {i} out of range for generated_projections_np")
                    else:
                        plt.close("all")  # Close any potentially open figures
                        fig = None  # Indicate no figure was created
                        buf = None
                        return

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(fig)

                buf.seek(0)

                # Send the raw bytes from the buffer via telegram
                if not self.disable_telegram:
                    try:
                        notifier.send_image(buf.getvalue(), caption=f"Epoch {epoch + 1} Generated Projections")
                    except Exception as e:
                        self.accelerator.print(f"Failed to send Telegram image: {e}")

                # close the buffer
                buf.close()
            except Exception as e:
                self.accelerator.print(f"Error creating visualization plot: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            self.accelerator.print(f"Error generating sample images: {e}")
            import traceback

            traceback.print_exc()


# Helper class for generating masks
class MaskGenerator3D:
    def __init__(self, args, num_channels=1):
        self.args = args
        self.num_channels = num_channels
        self.mask_type = getattr(self.args, "mask_type", "structured")  # Default if not in args

        # Initialize attributes based on self.args, applying defaults where necessary
        self.edge_type = getattr(self.args, "edge_type", None)
        self.edge_width = getattr(self.args, "edge_width", None)  # Used as proportion or None
        self.middle_axis = getattr(self.args, "middle_axis", None)

        # Defaults for randomization parameters if not provided in args
        self.middle_mask_width_min = getattr(self.args, "middle_mask_width_min", 0.08)
        self.middle_mask_width_max = getattr(self.args, "middle_mask_width_max", 0.15)
        self.middle_mask_position_jitter = getattr(self.args, "middle_mask_position_jitter", 0.05)

        # Conditional initialization based on mask_type
        if self.mask_type != "edge_mask":
            self.edge_type = None  # Only relevant for edge_mask
        if self.mask_type != "middle_mask":
            self.middle_axis = None  # Only relevant for middle_mask

        if self.middle_mask_width_min is not None and self.middle_mask_width_max is not None:
            if self.middle_mask_width_min > self.middle_mask_width_max:
                print(
                    f"Warning: middle_mask_width_min ({self.middle_mask_width_min}) > middle_mask_width_max ({self.middle_mask_width_max}). Swapping them."
                )
                self.middle_mask_width_min, self.middle_mask_width_max = (
                    self.middle_mask_width_max,
                    self.middle_mask_width_min,
                )
            if self.middle_mask_width_min == self.middle_mask_width_max:
                self.middle_mask_width_max += 0.01

    def __call__(self, shape):
        B, C_orig, D, H, W = shape
        mask = torch.zeros((B, self.num_channels, D, H, W))

        if self.mask_type == "middle_mask" and self.middle_axis is not None:
            current_width_ratio = None
            # Try to use randomized width range first
            if (
                self.middle_mask_width_min is not None
                and self.middle_mask_width_max is not None
                and self.middle_mask_width_min > 0
                and self.middle_mask_width_max > self.middle_mask_width_min
            ):  # Ensure min < max
                current_width_ratio = self.middle_mask_width_min + torch.rand(1).item() * (
                    self.middle_mask_width_max - self.middle_mask_width_min
                )
            # Fallback to self.edge_width if it's a valid proportion
            elif self.edge_width is not None and 0 < self.edge_width <= 1.0:
                current_width_ratio = self.edge_width
            # Else, it will use Dimension // 3 in the next step

            # Calculate actual bar widths based on the determined ratio or default
            d_bar_w = int(current_width_ratio * D) if current_width_ratio is not None else D // 3
            h_bar_w = int(current_width_ratio * H) if current_width_ratio is not None else H // 3
            w_bar_w = int(current_width_ratio * W) if current_width_ratio is not None else W // 3

            # Ensure minimum width of 1 pixel for the bar
            d_bar_w = max(1, d_bar_w)
            h_bar_w = max(1, h_bar_w)
            w_bar_w = max(1, w_bar_w)

            for b in range(B):
                if self.middle_axis == "depth":
                    actual_bar_width = d_bar_w
                    dim_size = D
                    base_start = (dim_size - actual_bar_width) // 2
                    max_offset = (
                        int(self.middle_mask_position_jitter * dim_size) if self.middle_mask_position_jitter > 0 else 0
                    )
                    offset = torch.randint(-max_offset, max_offset + 1, (1,)).item() if max_offset > 0 else 0
                    final_start = max(0, min(base_start + offset, dim_size - actual_bar_width))
                    mask[b, :, final_start : final_start + actual_bar_width, :, :] = 1
                elif self.middle_axis == "height":
                    actual_bar_width = h_bar_w
                    dim_size = H
                    base_start = (dim_size - actual_bar_width) // 2
                    max_offset = (
                        int(self.middle_mask_position_jitter * dim_size) if self.middle_mask_position_jitter > 0 else 0
                    )
                    offset = torch.randint(-max_offset, max_offset + 1, (1,)).item() if max_offset > 0 else 0
                    final_start = max(0, min(base_start + offset, dim_size - actual_bar_width))
                    mask[b, :, :, final_start : final_start + actual_bar_width, :] = 1
                elif self.middle_axis == "width":
                    actual_bar_width = w_bar_w
                    dim_size = W
                    base_start = (dim_size - actual_bar_width) // 2
                    max_offset = (
                        int(self.middle_mask_position_jitter * dim_size) if self.middle_mask_position_jitter > 0 else 0
                    )
                    offset = torch.randint(-max_offset, max_offset + 1, (1,)).item() if max_offset > 0 else 0
                    final_start = max(0, min(base_start + offset, dim_size - actual_bar_width))
                    mask[b, :, :, :, final_start : final_start + actual_bar_width] = 1
            return mask

        # Handle the edge masking case
        elif self.mask_type == "edge_mask" and self.edge_type is not None:
            # Calculate edge width if provided as proportion, otherwise use default
            d_width = self.edge_width * D if self.edge_width else D // 4
            h_width = self.edge_width * H if self.edge_width else H // 4
            w_width = self.edge_width * W if self.edge_width else W // 4

            # Convert to integers
            d_width = int(d_width)
            h_width = int(h_width)
            w_width = int(w_width)

            # Create edge mask based on specified edge
            for b in range(B):
                if self.edge_type == "right":
                    # Mask the right edge of the volume (along W)
                    mask[b, :, :, :, W - w_width :] = 1
                elif self.edge_type == "left":
                    # Mask the left edge of the volume (along W)
                    mask[b, :, :, :, :w_width] = 1
                elif self.edge_type == "top":
                    # Mask the top edge of the volume (along H)
                    mask[b, :, :, :h_width, :] = 1
                elif self.edge_type == "bottom":
                    # Mask the bottom edge of the volume (along H)
                    mask[b, :, :, H - h_width :, :] = 1
                elif self.edge_type == "front":
                    # Mask the front edge of the volume (along D)
                    mask[b, :, :d_width, :, :] = 1
                elif self.edge_type == "back":
                    # Mask the back edge of the volume (along D)
                    mask[b, :, D - d_width :, :, :] = 1

            return mask

        elif self.mask_type == "random_block":
            min_d, max_d = D // 2, int(D * 0.75)
            min_h, max_h = H // 2, int(H * 0.75)
            min_w, max_w = W // 2, int(W * 0.75)

            for b in range(B):
                # Random block dimensions
                block_d = torch.randint(min_d, max_d + 1, (1,)).item()
                block_h = torch.randint(min_h, max_h + 1, (1,)).item()
                block_w = torch.randint(min_w, max_w + 1, (1,)).item()

                # Random start position
                start_d = torch.randint(0, D - block_d + 1, (1,)).item()
                start_h = torch.randint(0, H - block_h + 1, (1,)).item()
                start_w = torch.randint(0, W - block_w + 1, (1,)).item()

                # Set mask region to 1 (unknown)
                mask[b, :, start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w] = 1

        elif self.mask_type == "multi_block":
            # Multiple smaller blocks, aiming for ~50% masked area
            num_blocks = torch.randint(self.args.min_blocks, self.args.max_blocks, (B,))  # 15-25 blocks per sample

            for b in range(B):
                for _ in range(num_blocks[b].item()):
                    # Larger block dimensions
                    block_d = torch.randint(D // 5, (D // 2) + 1, (1,)).item()  # Min D//5, Max D//2
                    block_h = torch.randint(H // 5, (H // 2) + 1, (1,)).item()  # Min H//5, Max H//2
                    block_w = torch.randint(W // 5, (W // 2) + 1, (1,)).item()  # Min W//5, Max W//2

                    # Random position
                    start_d = torch.randint(0, D - block_d + 1, (1,)).item()
                    start_h = torch.randint(0, H - block_h + 1, (1,)).item()
                    start_w = torch.randint(0, W - block_w + 1, (1,)).item()

                    mask[
                        b, :, start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w
                    ] = 1

        elif self.mask_type == "random_noise":
            # Random noise mask (more challenging)
            random_mask = torch.rand(B, 1, D, H, W) > 0.7  # 30% is unknown
            mask = random_mask.float()

        elif self.mask_type == "central_large_block":
            # Define relative size of the block (e.g., 40% to 70% of dimensions)
            min_ratio = getattr(self.args, "central_block_min_ratio", 0.4)
            max_ratio = getattr(self.args, "central_block_max_ratio", 0.7)
            # Ensure min_ratio is less than max_ratio
            if min_ratio >= max_ratio:
                min_ratio = max_ratio - 0.1  # Ensure some range
                if min_ratio < 0.1:
                    min_ratio = 0.1  # Ensure positive
                print(
                    f"Warning: central_block_min_ratio was >= max_ratio. Adjusted to min_ratio={min_ratio}, max_ratio={max_ratio}"
                )

            for b in range(B):
                # Random block dimensions as a ratio of the full dimension
                block_d_ratio = min_ratio + torch.rand(1).item() * (max_ratio - min_ratio)
                block_h_ratio = min_ratio + torch.rand(1).item() * (max_ratio - min_ratio)
                block_w_ratio = min_ratio + torch.rand(1).item() * (max_ratio - min_ratio)

                block_d = max(1, int(block_d_ratio * D))
                block_h = max(1, int(block_h_ratio * H))
                block_w = max(1, int(block_w_ratio * W))

                # Calculate start position to center the block, with slight jitter
                jitter_factor = getattr(self.args, "central_block_jitter_factor", 0.1)

                # Calculate the maximum possible offset for jitter
                # Jitter should not push the block outside the bounds if perfectly centered
                max_offset_d = int(jitter_factor * (D - block_d)) if (D - block_d) > 0 else 0
                max_offset_h = int(jitter_factor * (H - block_h)) if (H - block_h) > 0 else 0
                max_offset_w = int(jitter_factor * (W - block_w)) if (W - block_w) > 0 else 0

                offset_d = torch.randint(-max_offset_d, max_offset_d + 1, (1,)).item() if max_offset_d > 0 else 0
                offset_h = torch.randint(-max_offset_h, max_offset_h + 1, (1,)).item() if max_offset_h > 0 else 0
                offset_w = torch.randint(-max_offset_w, max_offset_w + 1, (1,)).item() if max_offset_w > 0 else 0

                # Ideal start for centered block
                center_start_d = (D - block_d) // 2
                center_start_h = (H - block_h) // 2
                center_start_w = (W - block_w) // 2

                # Apply jitter and ensure it stays within bounds
                start_d = max(0, min(D - block_d, center_start_d + offset_d))
                start_h = max(0, min(H - block_h, center_start_h + offset_h))
                start_w = max(0, min(W - block_w, center_start_w + offset_w))

                mask[b, :, start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w] = 1

        elif self.mask_type == "slice_mask":
            # Randomly mask slices along different axes
            axis = torch.randint(0, 3, (B,))  # 0=depth, 1=height, 2=width

            for b in range(B):
                if axis[b] == 0:  # Depth axis
                    # Mask random depth slices
                    num_slices = torch.randint(D // 4, D // 2, (1,)).item()
                    slice_indices = torch.randperm(D)[:num_slices]
                    for idx in slice_indices:
                        mask[b, :, idx, :, :] = 1
                elif axis[b] == 1:  # Height axis
                    # Mask random height slices
                    num_slices = torch.randint(H // 4, H // 2, (1,)).item()
                    slice_indices = torch.randperm(H)[:num_slices]
                    for idx in slice_indices:
                        mask[b, :, :, idx, :] = 1
                else:  # Width axis
                    # Mask random width slices
                    num_slices = torch.randint(W // 4, W // 2, (1,)).item()
                    slice_indices = torch.randperm(W)[:num_slices]
                    for idx in slice_indices:
                        mask[b, :, :, :, idx] = 1

        elif self.mask_type == "mixed":
            # Default: mix of different mask types
            mask_type_per_batch = torch.randint(0, 4, (B,))  # 0=block, 1=multi, 2=noise, 3=slice

            for b in range(B):
                if mask_type_per_batch[b] == 0:
                    # Single block
                    block_d = torch.randint(D // 4, D // 2, (1,)).item()
                    block_h = torch.randint(H // 4, H // 2, (1,)).item()
                    block_w = torch.randint(W // 4, W // 2, (1,)).item()
                    start_d = torch.randint(0, D - block_d + 1, (1,)).item()
                    start_h = torch.randint(0, H - block_h + 1, (1,)).item()
                    start_w = torch.randint(0, W - block_w + 1, (1,)).item()
                    mask[
                        b, :, start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w
                    ] = 1

                elif mask_type_per_batch[b] == 1:
                    # Multi-block
                    num_blocks = torch.randint(2, 5, (1,)).item()  # 2-4 blocks
                    for _ in range(num_blocks):
                        block_d = torch.randint(D // 8, D // 3, (1,)).item()
                        block_h = torch.randint(H // 8, H // 3, (1,)).item()
                        block_w = torch.randint(W // 8, W // 3, (1,)).item()
                        start_d = torch.randint(0, D - block_d + 1, (1,)).item()
                        start_h = torch.randint(0, H - block_h + 1, (1,)).item()
                        start_w = torch.randint(0, W - block_w + 1, (1,)).item()
                        mask[
                            b, :, start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w
                        ] = 1

                elif mask_type_per_batch[b] == 2:
                    # Random noise
                    random_mask = torch.rand(1, 1, D, H, W) > 0.7  # 30% is unknown
                    mask[b : b + 1] = random_mask.float()

                else:
                    # Slice mask
                    axis = torch.randint(0, 3, (1,)).item()  # 0=depth, 1=height, 2=width
                    if axis == 0:  # Depth
                        num_slices = torch.randint(D // 6, D // 3, (1,)).item()
                        slice_indices = torch.randperm(D)[:num_slices]
                        for idx in slice_indices:
                            mask[b, :, idx, :, :] = 1
                    elif axis == 1:
                        num_slices = torch.randint(H // 6, H // 3, (1,)).item()
                        slice_indices = torch.randperm(H)[:num_slices]
                        for idx in slice_indices:
                            mask[b, :, :, idx, :] = 1
                    else:  # Width
                        num_slices = torch.randint(W // 6, W // 3, (1,)).item()
                        slice_indices = torch.randperm(W)[:num_slices]
                        for idx in slice_indices:
                            mask[b, :, :, :, idx] = 1

        return mask
