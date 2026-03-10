"""
Filter Planktoscope images by ESD (Equivalent Spherical Diameter) size range.

Copies images within a specified ESD range to an output directory.
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd


def load_metadata(tsv_path: Path) -> pd.DataFrame:
    """Load EcoTaxa metadata, skipping type indicator row."""
    df = pd.read_csv(tsv_path, sep='\t', skiprows=[1])

    # Calculate ESD in microns
    pixel_size = df['process_pixel'].iloc[0]
    df['ESD_um'] = df['object_equivalent_diameter'] * pixel_size

    return df


def filter_and_copy_images(
    data_dir: Path,
    output_dir: Path,
    esd_min: float,
    esd_max: float,
    dry_run: bool = False
) -> int:
    """
    Filter images by ESD range and copy to output directory.

    Args:
        data_dir: Directory containing images and ecotaxa_export.tsv
        output_dir: Destination directory for filtered images
        esd_min: Minimum ESD in microns (inclusive)
        esd_max: Maximum ESD in microns (exclusive)
        dry_run: If True, only print what would be copied

    Returns:
        Number of images copied
    """
    tsv_path = data_dir / 'ecotaxa_export.tsv'
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    # Load and filter metadata
    df = load_metadata(tsv_path)
    filtered = df[(df['ESD_um'] >= esd_min) & (df['ESD_um'] < esd_max)]

    print(f"Found {len(filtered)} images with ESD in [{esd_min}, {esd_max}) µm")
    print(f"  (out of {len(df)} total images)")

    if len(filtered) == 0:
        return 0

    # Print ESD statistics for filtered images
    print(f"\nFiltered ESD statistics:")
    print(f"  Min: {filtered['ESD_um'].min():.1f} µm")
    print(f"  Max: {filtered['ESD_um'].max():.1f} µm")
    print(f"  Mean: {filtered['ESD_um'].mean():.1f} µm")

    if dry_run:
        print(f"\n[DRY RUN] Would copy {len(filtered)} images to {output_dir}")
        return len(filtered)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    copied = 0
    for _, row in filtered.iterrows():
        src = data_dir / row['img_file_name']
        dst = output_dir / row['img_file_name']

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  Warning: Image not found: {src}")

    print(f"\nCopied {copied} images to {output_dir}")

    # Save filtered metadata
    metadata_path = output_dir / 'filtered_metadata.csv'
    filtered.to_csv(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")

    return copied


def main():
    parser = argparse.ArgumentParser(
        description='Filter Planktoscope images by ESD size range'
    )
    parser.add_argument(
        'data_dir',
        type=Path,
        help='Directory containing images and ecotaxa_export.tsv'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory (default: data_dir/filtered_ESD_MIN-MAX)'
    )
    parser.add_argument(
        '--min', '-m',
        type=float,
        default=100.0,
        help='Minimum ESD in microns (default: 100)'
    )
    parser.add_argument(
        '--max', '-M',
        type=float,
        default=200.0,
        help='Maximum ESD in microns (default: 200)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be copied without copying'
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        args.output = args.data_dir / f'filtered_ESD_{args.min:.0f}-{args.max:.0f}'

    print(f"Filtering images from: {args.data_dir}")
    print(f"ESD range: [{args.min}, {args.max}) µm")
    print(f"Output directory: {args.output}")
    print()

    filter_and_copy_images(
        data_dir=args.data_dir,
        output_dir=args.output,
        esd_min=args.min,
        esd_max=args.max,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
