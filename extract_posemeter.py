#!/usr/bin/env python3
"""
Extract posemeter extension from NIRPS FITS files.

This script reads a FITS file containing multiple extensions and saves
only the posemeter extension to a designated output folder, preserving
both the primary header and the posemeter header.
"""

import argparse
import glob
from pathlib import Path
from typing import Optional
from astropy.io import fits

# Default output directory
OUTPUT_DIR = "/space/spirou/posemeter/"


def extract_posemeter(input_file: str, output_dir: str = OUTPUT_DIR) -> Optional[str]:
    """
    Extract posemeter extension from a FITS file and save it separately.

    Parameters
    ----------
    input_file : str
        Path to the input FITS file.
    output_dir : str
        Path to the output directory.

    Returns
    -------
    str or None
        Path to the output file, or None if skipped.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    # Generate output filename: original_name_posemeter.fits
    output_name = input_path.stem + "_posemeter.fits"
    output_file = output_path / output_name

    # Skip if output file already exists
    if output_file.exists():
        print(f"Skipping {input_path.name}: output file already exists")
        return None

    # Read the input FITS file
    with fits.open(input_file) as hdul:
        # Check if posemeter extension exists
        ext_names = [hdu.name.lower() for hdu in hdul]
        if 'posemeter' not in ext_names:
            print(f"Skipping {input_path.name}: no posemeter extension found")
            return None

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Get the primary header
        primary_header = hdul[0].header.copy()

        # Get the posemeter extension
        posemeter_hdu = hdul['posemeter']

        # Create new HDU list with primary header and posemeter extension
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        new_posemeter_hdu = fits.BinTableHDU(
            data=posemeter_hdu.data,
            header=posemeter_hdu.header.copy(),
            name='posemeter'
        )

        new_hdul = fits.HDUList([primary_hdu, new_posemeter_hdu])

        # Write to output file
        new_hdul.writeto(output_file, overwrite=True)

    print(f"Extracted posemeter extension to: {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Extract posemeter extension from NIRPS FITS files."
    )
    parser.add_argument(
        "input_pattern",
        help="Path to the input FITS file(s). Supports wildcards (e.g., '*.fits')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=OUTPUT_DIR,
        help=f"Path to the output directory (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()

    # Expand wildcard pattern
    files = sorted(glob.glob(args.input_pattern))

    if not files:
        print(f"No files found matching: {args.input_pattern}")
        return

    for input_file in files:
        extract_posemeter(input_file, args.output_dir)


if __name__ == "__main__":
    main()
