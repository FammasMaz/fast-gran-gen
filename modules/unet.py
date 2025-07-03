import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .conditioning import ConditioningEncoder
from typing import Optional


# Adapted from 2D and 3D unets from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish" or non_linearity == "silu":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None and self.time_emb_proj is not None:
            # broadcasting temb for 3D unet
            temb_reshaped = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
            hidden_states = hidden_states + temb_reshaped

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            # in our examples, transposed convolutions are not used since they introduce checkerboard artifacts
            self.conv = nn.ConvTranspose3d(channels, self.out_channels, kernel_size=2, stride=2)
        elif use_conv:
            self.conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # interpolation gave better results than transposed convolutions
        if output_size is None:
            # scale factor of 2 is used to match the downsampling block behavior
            output_size = (hidden_states.shape[2] * 2, hidden_states.shape[3] * 2, hidden_states.shape[4] * 2)

        hidden_states = F.interpolate(hidden_states, size=output_size, mode="trilinear", align_corners=False)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            # Use stride=2 for downsampling, also matched with the upsampling block interpolation behavior
            self.conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            # dynamic padding not yet implemented for downsampling
            raise NotImplementedError("Dynamic padding not yet implemented for Downsample3D")

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


#### Basic UNet Blocks (3D Adaptations, again from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets) ####


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        **kwargs,
    ):
        super().__init__()
        self.has_cross_attention = False  # no conditional attention for now
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            # no attention for now. easy to implement, but not used in our experiments
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(None)  # place holder for attention.

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # no attention for now. easy to implement, but not used in our experiments
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = False

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(None)  # place holder for attention.

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = False

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(None)  # place holder for attention.

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


#### UNet3DModel Definition ####


class UNet3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size=(32, 64, 64),
        in_channels=1,  # binary mask channel
        out_channels=1,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        mid_block_type="UNetMidBlock3D",
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(32, 64, 128, 256),  # set by default in the paper
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=None,  # not used for unconditional models we use
        attention_head_dim=8,  # match huggingface but not yet used
        # Add a flag to know if this model is for inpainting
        inpainting_mode: bool = False,
        # Conditioning parameters
        conditioning_dim: Optional[int] = None,  # dimension of conditioning features
        conditioning_hidden_dim: Optional[int] = None,  # hidden dim for conditioning encoder
        conditioning_dropout: float = 0.1,  # dropout for conditioning encoder
    ):
        super().__init__()

        self.sample_size = sample_size
        self._original_in_channels = in_channels
        self._original_out_channels = out_channels
        self.inpainting_mode = inpainting_mode

        # if inpainting mode is on: latent (C) + mask (1) + masked_latent (C) = 2*C + 1 reported in the paper
        conv_in_channels = (in_channels * 2 + 1) if inpainting_mode else in_channels

        time_embed_dim = block_out_channels[0] * 4

        # convolution layer for the input (dynamically set for inpainting or standard)
        self.conv_in = nn.Conv3d(conv_in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Conditioning encoder
        self.conditioning_dim = conditioning_dim
        if conditioning_dim is not None:
            self.conditioning_encoder = ConditioningEncoder(
                stats_dim=conditioning_dim,
                embed_dim=time_embed_dim,
                hidden_dim=conditioning_hidden_dim,
                dropout=conditioning_dropout
            )
        else:
            self.conditioning_encoder = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # custom 3d down
            down_block = DownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # custom 3d mid block
        if mid_block_type == "UNetMidBlock3D":
            self.mid_block = UNetMidBlock3D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
                num_layers=1,  # no further convolutions in the mid block except for the input to upsampling blocks
            )
        else:
            # Add more mid block types if needed later
            raise ValueError(f"Unsupported mid_block_type: {mid_block_type}")

        # custom 3d up blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            # custom 3d up block
            up_block = UpBlock3D(
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # final convolution to got back to out_channels which is 1 for binary mask.
        # TODO: alternatively test, outputing the same 3 channels and return the binary mask.
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,  # if inpainting mode, this is the concat [latent, mask, masked_latent]
        timestep: torch.LongTensor,
        conditioning_stats: Optional[torch.FloatTensor] = None,  # conditioning features
        return_dict: bool = True,
    ):
        # 1. Check inputs, inpainting or not.
        expected_channels = (
            (self._original_in_channels * 2 + 1) if self.inpainting_mode else self._original_in_channels
        )
        assert sample.dim() == 5, f"Input sample should be 5D, but got {sample.dim()} dimensions."
        assert sample.shape[1] == expected_channels, (
            f"Input sample has {sample.shape[1]} channels, but expected {expected_channels} for {'inpainting' if self.inpainting_mode else 'standard'} mode."
        )

        # [latent_zt (C), mask (1), masked_original_latent (C)]
        model_input = sample

        # considering that the input sample components are already correctly scaled [-1, 1]
        # TODO: make it configurable
        # if self.config.input_center_sample:
        #      pass

        # 2. Time embedding
        # Ensure timestep is properly formatted as 1D tensor and on the correct device
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif timestep.dim() == 0:  # scalar tensor
            timestep = timestep.unsqueeze(0)
        elif timestep.dim() > 1:  # multi-dimensional tensor, flatten to 1D
            timestep = timestep.flatten()

        # Ensure timestep is on the same device as the model
        timestep = timestep.to(sample.device)

        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        # 3. Conditioning embedding and combination
        if conditioning_stats is not None and self.conditioning_encoder is not None:
            # Validate conditioning input
            if conditioning_stats.dim() != 2:
                raise ValueError(f"conditioning_stats should be 2D tensor, got {conditioning_stats.dim()}D")
            if conditioning_stats.shape[0] != sample.shape[0]:
                raise ValueError(f"Batch size mismatch: conditioning_stats {conditioning_stats.shape[0]} vs sample {sample.shape[0]}")
            if conditioning_stats.shape[1] != self.conditioning_dim:
                raise ValueError(f"conditioning_stats feature dim {conditioning_stats.shape[1]} != expected {self.conditioning_dim}")
            
            # Encode conditioning features
            cond_emb = self.conditioning_encoder(conditioning_stats)
            
            # Combine time and conditioning embeddings
            emb = emb + cond_emb
        elif self.conditioning_encoder is not None and conditioning_stats is None:
            # Warning: conditioning encoder is available but no conditioning provided
            # This is expected during CFG when using null conditioning
            pass

        # 4. Initial convolution
        h = self.conv_in(model_input)  # pass the potentially concatenated input
        skip_connections = [h]

        # 5. Downsample
        for downblock in self.down_blocks:
            h, res_samples = downblock(h, emb)
            skip_connections.extend(res_samples)

        # 6. Mid-block (only one convolution)
        if self.mid_block is not None:
            h = self.mid_block(h, emb)

        # 7. Upsample (using interpolation)
        for upblock in self.up_blocks:
            res_samples = skip_connections[-len(upblock.resnets) :]  # Adjust skip connection count
            skip_connections = skip_connections[: -len(upblock.resnets)]
            h = upblock(h, res_samples, emb)

        # 8. Post-process
        h = self.conv_norm_out(h)  # group norm
        h = self.conv_act(h)  # silu activation
        sample = self.conv_out(h)  # final convolution to get binary mask

        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample

        if not return_dict:
            return (sample,)
        return UNetOutput(sample)
