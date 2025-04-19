├── .scratch
│   ├── data
|   |   ├── AGGC-2022-Classification
│   │   |   ├── test                   # Same structure as val sibling folder
│   │   |   ├── train                  # Same structure as val sibling folder
│   │   |   └── val
|   │   │       ├── g3                 # Folder holding patches - 100% of pixels belong to g3 class
|   │   │       ├── g4                 # Folder holding patches - 100% of pixels belong to g4 class
|   │   │       ├── g5                 # Folder holding patches - 100% of pixels belong to g5 class
|   │   │       ├── normal             # Folder holding patches - 100% of pixels belong to normal class
|   │   │       └── stroma             # Folder holding patches - 100% of pixels belong to stroma class
|   |   |           └── 00000.png      # Patch of size 512x512
|   |   ├── AGGC-2022-Patches
│   │   |   ├── test                       # Same structure as val sibling folder
│   │   |   ├── train                      # Same structure as val sibling folder
│   │   |   └── val
|   |   |       └── Subset1_Val_001
|   |   |           ├── 00001_g3.png       # 512x512 patch of g3 binary mask
|   |   |           ├── 00001_g4.png       # 512x512 patch of g4 binary mask
|   |   |           ├── 00001_g5.png       # 512x512 patch of g5 binary mask
|   |   |           ├── 00001_image.png    # 512x512 patch of WSI
|   |   |           ├── 00001_normal.png   # 512x512 patch of normal binary mask
|   |   |           └── 00001_stroma.png   # 512x512 patch of stroma binary mask
│   │   └── AGGC-2022-Unprepared
│   │       ├── test                       # Same structure as val sibling folder
│   │       ├── train                      # Same structure as val sibling folder
│   │       └── val
|   |           ├── Subset1_Val_001        # Folder with binary masks belonging to WSI with same name
|   |           |   ├── G3_Mask.tif        # [Optional] - Original Gleason 3 binary Mask
|   |           |   ├── G4_Mask.tif        # [Optional] - Original Gleason 4 binary Mask
|   |           |   ├── G5_Mask.tif        # [Optional] - Original Gleason 5 binary Mask
|   |           |   ├── Normal_Mask.tif    # [Optional] - Original Normal binary Mask
|   |           |   └── Stroma_Mask.tif    # [Optional] - Original Stroma 3 binary Mask
|   |           └── Subset1_Val_001.tiff   # Original WSI
│   ├── experiments
|   |   └── 0001 - Pretrained ResNet18 Grouped
|   |       ├── checkpoints
|   |       |   └── checkpoint-0000.pt   # StateDict of a model after 0th epoch
|   |       ├── config.yaml              # Configurations of model and its training
|   |       └── training.csv             # Metrics recorded during training
│   └── inference
|       └── Subset1_Val_001-512-0.5-0.5                  # <WSI-name>-<patch-size>-<overlap-percentage>-<tissue-coverage>
|           ├── classified_patches
|           |   └── 0001 - Pretrained ResNet18 Grouped
|           |       └── 40448_5120_normal.png            # <y-in-original-WSI>_<x-in-original-WSI>_<patch-type>.png   patch-types: gleason, normal, stroma
|           ├── finetuned_masks
|           |   └── 0001 - Pretrained ResNet18 Grouped
|           |       ├── Annotation.geojson               # Output finetuned masks for all 3 classes in geojson format
|           |       ├── Gleason_Mask.tif                 # Output finetuned mask for the Gleason class (G3&G4&G5) with size of the original WSI
|           |       ├── Normal_Mask.tif                  # Output finetuned mask for the Normal class with size of the original WSI
|           |       └── Stroma_Mask.tif                  # Output finetuned mask for the Stroma class with size of the original WSI
|           ├── masks
|           |   └── 0001 - Pretrained ResNet18 Grouped
|           |       ├── Gleason_Mask.tif                 # Output mask for the Gleason class (G3&G4&G5) with size of the original WSI
|           |       ├── Normal_Mask.tif                  # Output mask for the Normal class with size of the original WSI
|           |       └── Stroma_Mask.tif                  # Output mask for the Stroma class with size of the original WSI
|           ├── patches
|           |   └── 0_0_n.png                            # <y-in-original-WSI>_<x-in-original-WSI>_<patch-type>.png   patch-types: n-NoTissue, p-ProcessFurther
|           ├── visualizations
|           |   ├── 0001 - Pretrained ResNet18 Grouped
|           |   |   ├── gleason.png                      # Gleason class predictions overlayed over downsampled WSI
|           |   |   ├── gleason_with_gt.png              # Gleason class predictions overlayed over downsampled WSI with Fround Truth highlighted
|           |   |   ├── normal.png                       # Normal class predictions overlayed over downsampled WSI
|           |   |   ├── normal_with_gt.png               # Normal class predictions overlayed over downsampled WSI with Fround Truth highlighted
|           |   |   ├── stroma.png                       # Stroma class predictions overlayed over downsampled WSI
|           |   |   └── stroma_with_gt.png               # Stroma class predictions overlayed over downsampled WSI with Fround Truth highlighted
|           |   ├── downsampled.json                     # Holds the downsample scale-factor
|           |   ├── downsampled.png                      # Downsampled WSI
|           |   ├── downsampled_G3_Mask.png              # [Optional] Downsampled G3Mask
|           |   ├── downsampled_G4_Mask.png              # [Optional] Downsampled G4Mask
|           |   ├── downsampled_G5_Mask.png              # [Optional] Downsampled G4Mask
|           |   ├── downsampled_Normal_Mask.png          # [Optional] Downsampled NormalMask
|           |   ├── downsampled_Stroma_Mask.png          # [Optional] Downsampled StromaMask
|           |   ├── regions_to_process.png               # Downsampled WSI overlayed with regions selected for further processing
|           |   └── segmented.png                        # Downsampled binary mask for tissue segmentation
|           └── Subset1_Val_001_pyramidal.tiff           # The original WSI converted into a pyramidal representation for the QuPath program
├── project
│   ├── classification_training.py   # Script to run Classification Training
│   ├── batch_inference.py           # Script to perform inference over multiple WSIs and multiple models
│   └── inference.py                 # Script to perform complete inference over selected WSI using selected model
└── tools
    ├── classification               # Source code for Phase2: Classification
    ├── patching                     # Source code for Phase1: Patching
    ├── postprocessing               # Source code for Phase3: Postprocessing
    └── training                     # Source code for Classification Training