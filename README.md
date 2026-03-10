# Planktoscope Clustering #

## Goal/Methods ##

Create a pipeline to process data collected at the Santa Cruz Wharf and run through the Planktoscope.
`deep_feature_umap.ipynb` : This notebook clusters (HDBSCAN) samples based on the deep-features and UMAP embedding.
`explore_data.ipynb` : This notebook explores features based on the planktoscope segmentation software

## Data Structure ##

Data is organized by sample events where both collection methods are used:

```
data/
└── Santa-cruz-wharf_20260304/                    # Sample ID (location + date)
    └── Santa-cruz-wharf_20260304_PP/ # Net tow collection
    │   ├── *.jpg             # Segmented particle images
    │   └── ecotaxa_export.tsv
    └── Santa-cruz-wharf_20260304_WW/         # Whole Water
        ├── *.jpg
        └── ecotaxa_export.tsv
```

### EcoTaxa Metadata Format

The `ecotaxa_export.tsv` files contain 82 columns including:
- **Sample metadata**: project, sample_id, operator, lat/lon coordinates
- **Acquisition metadata**: instrument (PlanktoScope v2.6), camera settings, sampling date/time
- **Processing metadata**: 7-step segmentation pipeline parameters
- **Object features**: 60+ morphological measurements per particle (area, circularity, elongation, eccentricity, solidity, equivalent_diameter, hue/saturation/value statistics)

Key feature columns for analysis:
- `object_equivalent_diameter` - ESD in pixels (multiply by `process_pixel` for microns)
- `object_area` - particle area in pixels
- `object_circ.` - circularity (0-1)
- `object_elongation`, `object_eccentricity`, `object_solidity` - shape descriptors

### Pixel Size

The `process_pixel` column contains the pixel size (0.75 µm/pixel for current data).

## Required Packages
```
    - Pillow
    - torch
    - torchvision
    - pandas
    - numpy
    - matplotlib
    - umap-learn
    - hdbscan
    - tqdm
```