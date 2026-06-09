# Planktoscope Clustering Pipeline #

## Overview ##

This is an efficient way to explore image datasets, whether explore unlabeled data, or if validate labels.

This pipeline has follows this general workflow:

1. __HBDSCANFeature extractions__: This uses a deep-learning model to get a feature set from each image.
2. __UMAP Dimensional Reduction__: UMAP is non-linear dimensionality reduction method, that takes the deep features and reduces them to a few dimensions.
3. __HBDSCAN Clusteringtering__: (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a clustering method, with the main advantage is that you do not specify how many clusters should be.
4. __Manual Validation__: Review and download a cluseter of images to manually validate.
5.+ __Re-Clustering__: If cluster images appear mixed, taking a cluster and reclustering will help.

## Data Structure ##
The segmented images (rois) need to be in a flat file. I use the ecotaxa archive file (.zip) and extract it.

```
data/
└── Santa-cruz-wharf_20260304/ # Sample ID (location + date)
    └── Santa-cruz-wharf_20260304_PP/ # Net tow collection
    │   ├── *.jpg  # Segmented particle images
    │   └── ecotaxa_export.tsv  # Metadata and CV information (area, ESD, etc)
```

### EcoTaxa Metadata Format

The `ecotaxa_export.tsv` files contain 82 columns including:

- __Sample metadata__: project, sample_id, operator, lat/lon coordinates
- __Acquisition metadata__: instrument (PlanktoScope v2.6), camera settings, sampling date/time
- __Processing metadata__: 7-step segmentation pipeline parameters
- __Object features__: 60+ morphological measurements per particle (area, circularity, elongation, eccentricity, solidity, equivalent_diameter, hue/saturation/value statistics)

Key feature columns for analysis:

- `object_equivalent_diameter` - ESD in pixels (multiply by `process_pixel` for microns)
- `object_area` - particle area in pixels
- `object_circ.` - circularity (0-1)
- `:object_elongation`, `object_eccentricity`, `object_solidity` - shape descriptors

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

