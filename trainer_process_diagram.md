# Complete Process Flow: Trainer and generate_sample_images Function

## 1. Trainer Class Overview

```mermaid
graph TD
    A[Trainer.__init__] --> B[Setup Training Components]
    B --> C[Initialize Noise Scheduler]
    B --> D[Setup Optimizer & Learning Rate Scheduler]
    B --> E[Configure Accelerator/Device]
    B --> F[Initialize Logging WandB/TensorBoard]
    B --> G[Cache Context Samples for Inpainting]
    
    A --> H[train]
    H --> I[Training Loop]
    I --> J[train_epoch]
    I --> K[validate]
    I --> L[Log Reconstructions]
    I --> M[generate_sample_images]
    I --> N[Save Model & Telegram Updates]
```

## 2. Training Process Flow

```mermaid
graph LR
    A[Start Training] --> B[Initialize Components]
    B --> C[Training Loop for Each Epoch]
    C --> D[Train Step]
    D --> E[Add Noise to Clean Images]
    E --> F[Model Prediction]
    F --> G[Calculate Loss MSE/Weighted]
    G --> H[Backward Pass]
    H --> I[Validation Step]
    I --> J[Log Results Every 5 Epochs]
    J --> K[Generate Sample Images]
    K --> L[Save Best Model]
    L --> M{More Epochs?}
    M -->|Yes| C
    M -->|No| N[Training Complete]
```

## 3. generate_sample_images Function - Complete Flow

### 3.1 Initialization Phase

```mermaid
graph TD
    A[generate_sample_images epoch] --> B{Check if Main Process}
    B -->|No| C[Return None, None, None]
    B -->|Yes| D[Initialize Variables]
    D --> E[Get Model Config]
    E --> F[Determine Image Shape]
    F --> G{Check UNet Type}
    G -->|2D UNet| H[Shape: B,1,D,H,W where D=in_channels]
    G -->|3D UNet| I[Shape: B,C,D,H,W from sample_size]
```

### 3.2 Context Setup for Inpainting

```mermaid
graph TD
    A[Check Inpainting Mode] --> B{Is Inpainting UNet?}
    B -->|Yes| C[Use Cached Context Samples]
    C --> D[Adjust Sample Count to num_samples=4]
    D --> E[Initialize Noise with Seed]
    B -->|No| F[Standard Generation Mode]
    F --> E
```

### 3.3 Sampler Configuration

```mermaid
graph TD
    A[Initialize Sampler] --> B{Sampler Type}
    B -->|DDIM| C[Create DDIMScheduler]
    B -->|DDPM Default| D[Use Training Noise Scheduler]
    C --> E[Set Timesteps]
    D --> E
    E --> F[Get Timesteps Array]
```

### 3.4 Main Sampling Loop

```mermaid
graph TD
    A[Start Sampling Loop] --> B[For Each Timestep t]
    B --> C[Prepare Model Input]
    C --> D{Input Type}
    
    D -->|Inpainting| E[Generate/Use Mask]
    E --> F[Create Masked Images]
    F --> G[Concatenate: noisy_latents + mask + masked_images]
    
    D -->|2D UNet| H[Remove Channel Dim: B,D,H,W]
    
    D -->|3D Standard| I[Keep as B,C,D,H,W]
    
    G --> J[Model Forward Pass]
    H --> J
    I --> J
    
    J --> K[Get Noise Prediction]
    K --> L{Adapt Output}
    L -->|2D UNet| M[Add Channel Dim Back]
    L -->|3D UNet| N[Keep as is]
    
    M --> O[Scheduler Step]
    N --> O
    
    O --> P{Inpainting with Context?}
    P -->|Yes| Q[RePaint-like Step]
    Q --> R[Composite Known/Unknown Regions]
    P -->|No| S[Standard Step]
    
    R --> T{More Timesteps?}
    S --> T
    T -->|Yes| B
    T -->|No| U[Sampling Complete]
```

### 3.5 RePaint-like Inpainting Step (Detailed)

```mermaid
graph TD
    A[RePaint Step] --> B{Last Timestep?}
    B -->|No| C[Add Noise to Ground Truth]
    C --> D[Use Training Scheduler for Noise]
    D --> E[Composite: Generated×Mask + GT_Noisy×1-Mask]
    B -->|Yes| F[Final Composite: Generated×Mask + GT×1-Mask]
    E --> G[Continue to Next Step]
    F --> G
```

### 3.6 Post-Processing Pipeline

```mermaid
graph TD
    A[Convert to NumPy] --> B[Scale from -1,1 to 0,1]
    B --> C[Apply Gaussian Smoothing σ=0.5]
    C --> D[Binary Threshold > 0.5]
    D --> E[Connected Component Analysis]
    E --> F[Remove Small Components <1% volume]
    F --> G[Convert to Float32]
    G --> H[Store Processed Volumes]
```

### 3.7 Visualization and Saving

```mermaid
graph TD
    A[Post-Processing Complete] --> B{PyVista Available?}
    B -->|Yes| C[Save VTI Files]
    C --> D[Create Temporary Directory]
    D --> E[Save Inpainted VTIs]
    E --> F{Has Original Context?}
    F -->|Yes| G[Save Original VTIs]
    G --> H[Save Masked VTIs]
    F -->|No| H
    H --> I[Compress to TAR.GZ]
    I --> J[Clean Temp Directory]
    
    B -->|No| K[Skip VTI Saving]
    J --> K
    K --> L[Create Visualizations]
```

### 3.8 Visualization Creation

```mermaid
graph TD
    A[Create Visualizations] --> B[Extract Middle Depth Slice]
    B --> C[Create Max Projections]
    C --> D{Inpainting Mode?}
    D -->|Yes| E[Create Masked Input Visualization]
    E --> F[Find Best Slice with Most Mask Content]
    F --> G[Create RGB Colored Context]
    G --> H[Apply Mask Overlay]
    H --> I[Side-by-side: Masked + Raw + Processed]
    D -->|No| J[Standard Projections Only]
    I --> K[Log to WandB/TensorBoard]
    J --> K
    K --> L[Create Matplotlib Figure]
    L --> M[Send via Telegram]
```

## 4. Key Data Structures and Transformations

### 4.1 Tensor Shapes Throughout Process

| Stage | 2D UNet Shape | 3D UNet Shape | Inpainting Shape |
|-------|---------------|---------------|------------------|
| Initial | `(B,1,D,H,W)` | `(B,C,D,H,W)` | `(B,C,D,H,W)` |
| Model Input | `(B,D,H,W)` | `(B,C,D,H,W)` | `(B,2C+1,D,H,W)` |
| Model Output | `(B,D,H,W)` | `(B,C,D,H,W)` | `(B,C,D,H,W)` |
| Final | `(B,1,D,H,W)` | `(B,C,D,H,W)` | `(B,C,D,H,W)` |

### 4.2 Mask Generation Types

```mermaid
graph LR
    A[MaskGenerator3D] --> B[middle_mask]
    A --> C[edge_mask]
    A --> D[random_block]
    A --> E[multi_block]
    A --> F[central_large_block]
    A --> G[mixed_edge_central]
    A --> H[gap_filling_compatible]
    A --> I[slice_mask]
    A --> J[random_noise]
```

## 5. Loss Calculation Variants

```mermaid
graph TD
    A[Calculate Loss] --> B{Use Weighted Loss?}
    B -->|No| C[Standard MSE Loss]
    B -->|Yes| D{Inpainting Mode?}
    D -->|No| C
    D -->|Yes| E[Create Weight Mask]
    E --> F[Higher Weight for Masked Regions]
    F --> G[Weighted MSE Loss]
    C --> H[Backward Pass]
    G --> H
```

## 6. Function Return Values

The `generate_sample_images` function returns a tuple of three elements:

1. **`processed_volumes_01`**: List of processed inpainted volumes in [0,1] range
2. **`original_vti_processed`**: List of processed original context volumes (inpainting mode only)
3. **`masked_vti_processed`**: List of processed masked context volumes (inpainting mode only)

For non-inpainting modes, the last two return values are `None`.

## 7. Key Features

### 7.1 Inpainting Capabilities
- Uses cached context samples from training data
- Supports multiple mask types via `MaskGenerator3D`
- Implements RePaint-like guidance during sampling
- Creates side-by-side visualizations of input/output

### 7.2 Multi-Modal Support
- **2D UNet**: Treats depth as channels `(B,D,H,W)`
- **3D UNet**: Standard volumetric processing `(B,C,D,H,W)`
- **Inpainting UNet**: Concatenated input `[noisy, mask, masked_context]`

### 7.3 Robust Sampling
- Supports both DDPM and DDIM samplers
- Configurable number of inference steps
- Deterministic seeding for reproducibility
- GPU memory efficient processing

### 7.4 Advanced Post-Processing
- Gaussian smoothing for noise reduction
- Connected component filtering
- Binary thresholding for clean outputs
- Multiple visualization formats (slices, projections)

This comprehensive pipeline enables the trainer to generate high-quality 3D volumetric samples with support for both standard generation and sophisticated inpainting tasks.