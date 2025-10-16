# Gemini Project Overview: SpatialVLA

## Project Overview

This project, **SpatialVLA**, is a vision-language-action model designed for robotics applications. It enhances traditional VLA models with spatial representations, enabling more accurate and efficient robot manipulation. The model is built using Python and leverages the HuggingFace Transformers library, making it easy to use and extend. The core of SpatialVLA is based on the PaLiGemma2 architecture, and it uses a ZoeDepth model for depth estimation.

## New Research Direction: Hybrid Geometric-Semantic Vision

Our current research objective is to re-architect the model's vision pipeline to use a hybrid approach, combining separate geometric and semantic features.

-   **Geometric Features**: We will integrate the **Map-Anything** model to generate high-quality 3D geometric representations (e.g., point clouds) of the scene.
-   **Semantic Features**: We will retain the existing **SigLIP** model (`vision_tower`) to provide rich 2D semantic features.

This dual-branch approach will provide the model with both the "what" (semantics) and the "where" (geometry) of the scene. Future work will involve experimenting with other semantic encoders, such as DINOv2.

### Integration Plan:

1.  **Create Parallel Vision Branches (`model/modeling_spatialvla.py`):**
    *   In `SpatialVLAVisionModel.__init__`, we will **keep** the existing `vision_tower` (SigLIP). The old depth-based 3D pipeline (`depth_model`, `pos_embed_3d`) will be removed.
    *   A new branch will be added to initialize the **Map-Anything** model, loading it from the Hugging Face Hub.

2.  **Develop a Fusion Module (`model/modeling_spatialvla.py`):**
    *   The core of the new work will be in the `SpatialVLAVisionModel.forward` method.
    *   This method will now process the input image through both the SigLIP and Map-Anything branches in parallel.
    *   A new **fusion mechanism** must be developed to combine the 2D semantic features from SigLIP with the 3D geometric features from Map-Anything before they are passed to the language model.

3.  **Update Configuration (`model/configuration_spatialvla.py`):**
    *   The `SpatialVLAConfig` class will be modified to include new parameters for configuring the Map-Anything model (e.g., its Hugging Face model identifier).

## Development Principles

- **Transparency**: For every code modification or design decision, I will provide a clear and concise explanation of the 'what' and the 'why' to ensure our collaboration is transparent and avoids confusion.

## Building and Running

### Dependencies

The project's dependencies are listed in `pyproject.toml` and `requirements.txt`. Key dependencies include:

*   `torch`
*   `transformers`
*   `tensorflow`
*   `peft` (for LoRA fine-tuning)
*   `deepspeed` (for distributed training)

### Pre-training

To pre-train the model from scratch, you can use the `torchrun_pretrain.sh` script:

```bash
bash scripts/spatialvla_4b_pretrain/torchrun_pretrain.sh
```

This script handles the distributed training setup and launches the `train/spatialvla_pretrain.py` script with the appropriate arguments. You will need to download the Open X-Embodiment and RH20T datasets and configure the data paths in the script.

### Fine-tuning

For fine-tuning, the project supports both full-parameter and LoRA fine-tuning. The `finetune_lora.sh` script is provided for LoRA fine-tuning:

```bash
bash scripts/spatialvla_4b_finetune/finetune_lora.sh
```

This script uses the `train/spatialvla_finetune.py` script to fine-tune the model on a smaller dataset. You can customize the LoRA rank, alpha, and other hyperparameters in the script.

## Development Conventions

*   **Model Architecture**: The core model is defined in `model/modeling_spatialvla.py`. It integrates a vision tower, a language model, and a multi-modal projector. The model uses a custom `SpatialVLAPreTrainedModel` class that supports gradient checkpointing and other advanced features.
*   **Training**: The training scripts are located in the `train/` directory. They use the HuggingFace Trainer API and support distributed training with DeepSpeed.
*   **Data**: The `data/` directory contains scripts for data loading and processing. The project uses the RLDS format for datasets.
*   **Configuration**: The model and training configurations are managed through a combination of shell scripts and Python dataclasses. Key configurations are stored in JSON files in the `scripts/` directory.
*   **Action Tokenization**: The project uses a custom `SpatialActionTokenizer` to convert robot actions into a sequence of tokens that can be processed by the language model.
*   **Testing**: A `test_huggingface.py` file is provided in the `test/` directory to verify the model's integration with the HuggingFace ecosystem.