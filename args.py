import argparse
from pathlib import Path


# convert string to boolean for easier debugging
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    This function is used to get the arguments for the training script. This global argument parser defaults to the values that
    were used for the final experiments in the paper for reproducibility purposes, except for the strings for dataset paths,
    which should be set by the user
    """
    parser = argparse.ArgumentParser()

    ##### Dataset configuration and training parameters #####
    parser.add_argument(
        "--root_dir", type=str, default="dataset/graph_imgs_tiled/"
    )  # this path should point to root dataset dir in the zip file
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="dataset/.cache",
        help="Directory to store dataset cache files",
    )  # this is the directory to store the dataset cache files
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--img_size", type=tuple, default=(512, 512)
    )  # this is the original 2D grid size used to convert mesh into grids.
    parser.add_argument("--timesteps", type=int, default=1000)  # noise scheduling timesteps
    parser.add_argument("--epochs", type=int, default=160)  # number of epochs to train for
    parser.add_argument(
        "--accelerator", type=str2bool, default=True
    )  # prepares the code for distributed/mixed precision training
    parser.add_argument(
        "--device", type=str, default="cuda"
    )  # default device to use where accelerator is not available
    parser.add_argument(
        "--transforms", type=str2bool, default=False
    )  # TODO:currently not used, but can be used to apply random transformations
    parser.add_argument("--mixed_precision", type=str2bool, default=True)  # use mixed precision training
    parser.add_argument(
        "--small", type=str2bool, default=False
    )  # use smaller dataset for testing/dataset debugging only, else false
    parser.add_argument(
        "--patience", type=int, default=500
    )  # number of epochs to wait before stopping training if no improvement, currently not used (arbitarily high)
    parser.add_argument("--output_dir", type=str, default="out/")  # directory to save the model checkpoints and logs
    parser.add_argument(
        "--with_metadata", type=str2bool, default=False
    )  # TODO: not used, but can be used to save metadata
    parser.add_argument("--scheduler", type=str, default="squaredcos_cap_v2")  # noise scheduling function
    parser.add_argument(
        "--patched", action="store_true"
    )  # TODO: used for training on huge datasamples and not on patches, code ommitted for brevity and non-relevance
    parser.add_argument(
        "--bw_ratio", type=float, default=0.2
    )  # minimum number of 1s in the binary mask for voxel grids
    parser.add_argument(
        "--only_mse", action="store_true", help="Only use MSE loss"
    )  # use only mse loss in various places in code
    parser.add_argument("--check_for_edges", action="store_true")  # check for edges in the voxel grids
    parser.add_argument("--diffusion_lr", type=float, default=1e-4)  # learning rate for the diffusion model
    parser.add_argument("--num_workers", type=int, default=4)  # number of workers for the dataloader
    parser.add_argument(
        "--percentage", type=float, default=0.99
    )  # percentage of the dataset to use for training (arbitrarily full dataset)
    parser.add_argument("--sdf_scale", type=float, default=5.0)  # scale for the sdf
    parser.add_argument("--seed", type=int, default=42)  # seed for the random number generator
    parser.add_argument(
        "--use_sdf", type=str2bool, default=False
    )  # dont use sdf for now. we found that training with sdf was not better than training with binary data.
    parser.add_argument("--checkpoint_epoch", type=int, default=None, help="Checkpoint every N epochs.")
    parser.add_argument(
        "--sampler_type",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Type of sampler to use for inpainting.",
    )  # ddim is recommended for inpainting. much faster, with same quality.
    parser.add_argument(
        "--use_2d_unet", action="store_true", help="Use a 2D UNet treating depth as channels instead of a 3D UNet."
    )  # not recommended. 2D UNet is not as good as 3D UNet.

    ##### EMA (Exponential Moving Average) Configuration #####
    parser.add_argument(
        "--use_ema", type=str2bool, default=False, help="Use Exponential Moving Average for model weights."
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay rate (higher values give more smoothing)."
    )
    parser.add_argument(
        "--ema_update_after_step", type=int, default=100, help="Start EMA updates after this many training steps."
    )
    parser.add_argument("--ema_update_every", type=int, default=1, help="Update EMA every N training steps.")
    parser.add_argument(
        "--use_ema_for_validation", type=str2bool, default=True, help="Use EMA weights for validation."
    )
    parser.add_argument(
        "--use_ema_for_generation", type=str2bool, default=True, help="Use EMA weights for sample generation."
    )
    parser.add_argument(
        "--save_ema_as_final", type=str2bool, default=True, help="Save EMA weights as the final model."
    )

    ##### Inpainting Mode and Mask Configuration #####

    parser.add_argument(
        "--inpainting_mode", type=str2bool, default=False
    )  # switch between unconditional and inpainting training. Default is unconditional.
    parser.add_argument("--repaint_guidance", action="store_true", help="Use repaint guidance for inpainting.")
    parser.add_argument("--use_weighted_loss", action="store_true", help="Use weighted loss for inpainting.")
    # many different types of masks are used for inpainting. The results in the paper are obtained using the central_large_block mask.
    parser.add_argument(
        "--mask_type",
        type=str,
        default="central_large_block",
        choices=[
            "random_block",
            "multi_block",
            "random_noise",
            "slice_mask",
            "mixed",
            "edge_mask",
            "middle_mask",
            "central_large_block",
            "mixed_edge_central",
            "gap_filling_compatible",
        ],
        help="Type of mask to use for inpainting. Central large block is the default used in NeurIPS paper. mixed_edge_central combines central blocks with edge masks for better junction inpainting. gap_filling_compatible creates thin strips for gap-filling tasks.",
    )
    parser.add_argument(
        "--central_block_max_ratio",
        type=float,
        default=0.8,
        help="Maximum ratio of the central block size to the total dimension.",
    )  # only used if mask_type is central_large_block. Default value was 0.7. Higher values were not yet tested.
    parser.add_argument(
        "--central_block_min_ratio",
        type=float,
        default=0.7,
        help="Minimum ratio of the central block size to the total dimension.",
    )
    parser.add_argument(
        "--edge_type",
        type=str,
        default="right",
        choices=["right", "left", "top", "bottom", "front", "back"],
        help="Edge to mask when using edge_mask type",
    )  # only used if mask_type is edge_mask
    parser.add_argument(
        "--edge_width", type=float, default=0.2, help="Width of masked region as a proportion of dimension"
    )  # only used if mask_type is edge_mask
    parser.add_argument(
        "--middle_axis",
        type=str,
        default="depth",
        choices=["depth", "height", "width"],
        help="Axis to mask through the middle when using middle_mask type",
    )  # only used if mask_type is middle_mask
    parser.add_argument(
        "--middle_mask_width_min",
        type=float,
        default=None,  # handled by MaskGenerator3D in code
        help="Minimum width of the middle mask bar as a proportion of dimension (e.g., 0.08). Used if mask_type is middle_mask.",
    )
    parser.add_argument(
        "--middle_mask_width_max",
        type=float,
        default=None,  # handled by MaskGenerator3D in code
        help="Maximum width of the middle mask bar as a proportion of dimension (e.g., 0.15). Used if mask_type is middle_mask.",
    )
    parser.add_argument(
        "--middle_mask_position_jitter",
        type=float,
        default=None,  # handled by MaskGenerator3D in code
        help="Maximum positional jitter for the middle mask bar as a proportion of dimension (e.g., 0.05 for +/-5%%). Used if mask_type is middle_mask.",
    )
    # if using multiblock set min and max blocks
    parser.add_argument("--min_blocks", type=int, default=8, help="Minimum number of blocks to use for inpainting.")
    parser.add_argument("--max_blocks", type=int, default=15, help="Maximum number of blocks to use for inpainting.")

    #### Logger configuration ####

    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",  # Default to using wandb
        choices=["wandb", "tensorboard", "none"],
        help="Logging backend to use. Wandb is recommended for better visualization.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",  # server used didnt have internet on compute nodes, hence wandb webhooks were used to update
        choices=["online", "offline"],
        help="WandB mode (online or offline). Only used if --logger is wandb.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="diffusion-model-3d",
        help="WandB project name (or general project identifier).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="WandB run name.",
    )

    parser.add_argument(
        "--disable_telegram",
        action="store_true",
        help="Disable all Telegram notifications and image sending.",
    )  # a boolean flag to disable all Telegram notifications and image sending. easier to debug on the go.

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.model_dir = args.output_dir / "diffusion_model"  # directory to save the model checkpoints and logs by default

    return args
