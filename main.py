import warnings
from args import get_args
from modules.trainer import Trainer
from dataloader import get_voxel_dataloaders
from modules.unet import UNet3DModel
from diffusers import UNet2DModel
from utils.telegram_notifier import notifier
from utils.device_utils import get_system_info, get_device_count
import platform
import torch
import datetime
import sys


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    args = get_args()

    # some system debug info
    system_info = get_system_info(args.device)

    # send initial notification with system info via TG
    notifier.send_message(
        f" 3D Diffusion Training Run Initialized\n"
        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"System: {system_info['system']} {system_info['release']}\n"
        f"Python: {system_info['python']}, PyTorch: {system_info['torch']}\n"
        f"Device: {system_info['device_name']}"
    )

    # ensure batch size is reasonable for the number of accelerators
    n_devices = get_device_count(args.device)
    if n_devices > 1 and args.batch_size < n_devices * 2:
        print(f"Warning: Batch size {args.batch_size} may be too small for {n_devices} GPUs")
        print(f"Consider increasing batch size to at least {n_devices * 2}")

    train_loader, val_loader = get_voxel_dataloaders(
        h5_file_path=args.root_dir,  # path to the h5 file
        batch_size=args.batch_size,  # batch size for training
        shuffle=True,  # shuffle the data
        num_workers=args.num_workers,  # number of workers for loading data
        transform=None,  # transform the data, currently not used
        small=args.small,  # use small dataset or not
        val_split=0.2,  # validation split
        bw_ratio=args.bw_ratio,  # background ratio
        check_for_edges=args.check_for_edges,
        edge_thickness=2,
        pin_memory=True,
        drop_last=True,
        cache_size=5000,
        mapping_cache_dir=args.cache_dir,
        percentage=args.percentage,
        sdf=args.use_sdf,
        interpol=False,  # interpolation is not used
        sdf_scale=args.sdf_scale,
    )

    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {sample_batch.shape}")

    if sample_batch.dim() != 5:
        raise ValueError(f"Expected 5D tensor from dataloader, but got shape {sample_batch.shape}")
    if sample_batch.shape[1] != 1:
        print(
            f"Warning: Expected input channels (C) to be 1, but got {sample_batch.shape[1]}. Adjusting model config."
        )

    _, C_in, D, H, W = sample_batch.shape
    depth_channels = D
    sample_size_3d = (D, H, W)
    sample_size_2d = (H, W)
    print(f"3D Sample size (D, H, W): {sample_size_3d}")
    print(f"2D Sample size (H, W): {sample_size_2d}")
    print(f"Depth/Channels for 2D UNet: {depth_channels}")

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    notifier.send_message(
        f"📊 3D Dataset Loaded\n"
        f"Train samples: {train_size}\n"
        f"Val samples: {val_size}\n"
        f"Batch shape: {sample_batch.shape}\n"
        f"Sample size (D, H, W): {sample_size_3d}\n"
    )

    if args.use_2d_unet:
        print("Using 2D UNet (Depth as Channels)")
        model = UNet2DModel(
            sample_size=sample_size_2d,
            in_channels=depth_channels,
            out_channels=depth_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 1024),  # usually requires a denser network
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            act_fn="silu",
            norm_num_groups=32,
        )
        model_type_str = f"UNet2DModel (Depth={depth_channels} as Channels)"
        model_input_shape_str = f"({depth_channels}, {H}, {W})"

    else:
        print("Using 3D UNet for Inpainting")
        model = UNet3DModel(
            sample_size=sample_size_3d,
            in_channels=C_in,
            out_channels=C_in,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
            ),
            mid_block_type="UNetMidBlock3D",
            act_fn="silu",
            norm_num_groups=32,
            inpainting_mode=args.inpainting_mode,
        )
        model_type_str = "UNet3DModel (Inpainting)"
        inpainting_channels = C_in * 2 + 1
        model_input_shape_str = f"({inpainting_channels}, {D}, {H}, {W})"

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: {model_type_str}")
    print(f"Model Parameters: {model_params:,}")
    print(f"Size of the model: {model_params * 4 / 1024 / 1024} MB")
    notifier.send_message(
        f"Model Architecture\n"
        f"Type: {model_type_str}\n"
        f"Parameters: {model_params:,}\n"
        f"Model Input Shape (C, D, H, W or C, H, W): {model_input_shape_str}\n"
    )

    # show what is the mask configuration
    if args.inpainting_mode:
        print("Inpainting mode enabled with mask configuration:")
        print(f"  - Mask Type: {args.mask_type}")
        if args.mask_type == "edge_mask":
            print(f"  - Edge Type: {args.edge_type}")
            print(f"  - Edge Width: {args.edge_width}")

        mask_info = f"Inpainting with {args.mask_type}"
        if args.mask_type == "edge_mask":
            mask_info += f" ({args.edge_type} edge, width={args.edge_width})"

        notifier.send_message(f"Inpainting Configuration\n{mask_info}")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        args,
        dataset=train_loader.dataset.dataset,
        use_2d_unet=args.use_2d_unet,
        disable_telegram=args.disable_telegram,
    )

    try:
        trainer.train()
        notifier.send_message("Training completed successfully!")
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        notifier.send_message(f"Training failed\nError: {str(e)}\nTrace: {error_trace[:500]}")
        print(f"Error occurred: {str(e)}")
        print(error_trace)
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing

    # macOS (Apple Silicon / MPS) requires "spawn"; Linux can use "fork"
    start_method = "spawn" if platform.system() == "Darwin" else "fork"
    try:
        multiprocessing.set_start_method(start_method)
    except RuntimeError:
        pass
    main()
