#!/usr/bin/env python3
"""
=============================================================================
NIRPS Posemeter Analysis Tool
=============================================================================

This script analyzes posemeter data extracted from NIRPS FITS files to assess
the stability of fiber-to-fiber flux ratios during observations.

The posemeter measures flux in two fibers (FIBRE1 and FIBRE2) throughout an
exposure. By computing the normalized difference between fibers, we can detect:
- Guiding issues (large RMS in the difference)
- Cloud passages (systematic trends)
- Instrument instabilities

Key metrics computed:
- MED: Median of (FIBRE1 - FIBRE2), proxy for total flux
- RMS: Standard deviation of normalized difference, proxy for stability

Output:
- Summary plot: RMS vs Date, color-coded by flux level
- Offender plot: Detailed view of high-RMS observations

Author: NIRPS Team
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
import numpy as np
import glob
import os
import csv
import yaml
import matplotlib.dates as mdates
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the reject list CSV file (files to skip on future runs)
REJECT_FILE = 'data/reject_list.csv'

# Directory to save output plots
PLOT_DIR = 'plots'

# Object keywords to reject (calibration files, not science targets)
BAD_OBJECT_KEYWORDS = ['FLAT', 'WAVE', 'ORDERDEF', 'CONTAM', 'FP', 'SKY', 'TELLURIC','LED','DARK','sky']

# =============================================================================
# LOAD CONFIG FILE
# =============================================================================

CONFIG_FILE = 'config.yaml'
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as _f:
        _cfg = yaml.safe_load(_f)
    print(f"Loaded configuration from {CONFIG_FILE}")
else:
    _cfg = {}
    print(f"No config file found ({CONFIG_FILE}), using defaults")

# List of objects to keep; empty list means keep all science objects
OBJECT_FILTER = [str(o) for o in _cfg.get('objects', [])]

# Whether to discard the first data point of each posemeter time series
REJECT_FIRST_POINT = bool(_cfg.get('reject_first_point', True))

# Flux and RMS thresholds
MIN_FLUX_THRESHOLD = float(_cfg.get('min_flux_threshold', 2))
HIGH_RMS_THRESHOLD = float(_cfg.get('high_rms_threshold', 0.1))

if OBJECT_FILTER:
    print(f"Object filter active: {OBJECT_FILTER}")
else:
    print("No object filter – all science objects will be processed")
print(f"Reject first point: {REJECT_FIRST_POINT}")

# =============================================================================
# LOAD REJECT LIST
# =============================================================================
# The reject list contains files that should be permanently skipped
# (e.g., calibration files identified in previous runs)

if os.path.exists(REJECT_FILE):
    with open(REJECT_FILE, 'r') as f:
        reject_list = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(reject_list)} files from reject list")
else:
    reject_list = set()
    print("No reject list found, starting fresh")

# =============================================================================
# FILE INDEX (CSV CACHE)
# =============================================================================
# The index stores one row per FITS file: filename, object, mjd
# It is built incrementally — only new files trigger header reads.
# This allows fast per-object filtering without opening every FITS file.

INDEX_FILE = 'data/file_index.csv'
INDEX_FIELDS = ['filename', 'object', 'mjd']

# Load existing index
index = {}  # basename -> {'object': ..., 'mjd': ...}
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, 'r', newline='') as _fh:
        for row in csv.DictReader(_fh):
            index[row['filename']] = {'object': row['object'], 'mjd': float(row['mjd'])}
    print(f"Loaded index with {len(index)} entries from {INDEX_FILE}")
else:
    print(f"No index file found — will build {INDEX_FILE}")

# Find all FITS files on disk
all_files = sorted(glob.glob('data/NIRPS_*_posemeter.fits'))
print(f"Found {len(all_files)} posemeter files on disk")

# Identify files missing from the index
new_files = [f for f in all_files if os.path.basename(f) not in index]
if new_files:
    print(f"Indexing {len(new_files)} new file(s)...")
    new_rows = []
    for f in new_files:
        try:
            hdr = fits.getheader(f)
            entry = {'object': hdr.get('OBJECT', 'UNKNOWN'), 'mjd': float(hdr.get('MJD-OBS', 0))}
            index[os.path.basename(f)] = entry
            new_rows.append(entry | {'filename': os.path.basename(f)})
        except Exception as exc:
            print(f"  WARNING: could not read {f}: {exc}")
    # Append new rows to CSV (create with header if needed)
    write_header = not os.path.exists(INDEX_FILE)
    with open(INDEX_FILE, 'a', newline='') as _fh:
        writer = csv.DictWriter(_fh, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"Index updated — {len(new_rows)} row(s) added to {INDEX_FILE}")
else:
    print("Index is up to date")

# =============================================================================
# FIND AND SORT INPUT FILES  (using index for fast filtering)
# =============================================================================
# Track newly identified calibration files to add to reject list
new_rejects = []

# Build the filtered file list entirely from the in-memory index — no FITS I/O
files = []
for f in all_files:
    bn = os.path.basename(f)
    if bn in reject_list:
        continue
    object_name = index.get(bn, {}).get('object', '')
    if any(bad_obj in object_name for bad_obj in BAD_OBJECT_KEYWORDS):
        new_rejects.append(bn)
        continue
    if OBJECT_FILTER and object_name not in OBJECT_FILTER:
        continue
    files.append(f)

# Persist any newly identified calibration files
if new_rejects:
    with open(REJECT_FILE, 'a') as _fh:
        for bn in new_rejects:
            _fh.write(bn + '\n')
    reject_list.update(new_rejects)
    print(f"Added {len(new_rejects)} calibration file(s) to reject list")

print(f"{len(files)} file(s) pass the object filter and will be analysed")

# =============================================================================
# INITIALIZE DATA STORAGE
# =============================================================================

# Vectors to store computed metrics for each valid file
rms_vec = []      # RMS of normalized fiber difference (stability metric)
med_vec = []      # Median flux difference (brightness metric)
mjd_vec = []      # Modified Julian Date of observation
file_vec = []     # Filename for reference
ks_stat_vec = []  # KS D-statistic: first-half vs second-half of normalized diff (0=same, 1=fully separated)

# Store detailed data for high-RMS files (for later plotting)
high_rms_data = []

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

print("\nProcessing files...")
print("-" * 60)

total_files = len(files)
for i_file, file in enumerate(files, 1):
    object_name = index[os.path.basename(file)]['object']
    print(f"[{i_file}/{total_files}] Processing: {object_name}")

    # -------------------------------------------------------------------------
    # Read posemeter table from FITS extension
    # Columns: TIME, FIBRE1, FIBRE2
    # -------------------------------------------------------------------------
    tbl = Table.read(file, hdu='posemeter')

    # -------------------------------------------------------------------------
    # Optionally discard the first data point
    # -------------------------------------------------------------------------
    if REJECT_FIRST_POINT and len(tbl) > 1:
        tbl = tbl[1:]

    # -------------------------------------------------------------------------
    # Compute fiber difference and normalize
    # -------------------------------------------------------------------------
    # The difference (FIBRE1 - FIBRE2) removes common-mode variations
    # Normalizing by median gives relative variations (dimensionless)
    diff = tbl['FIBRE1'] - tbl['FIBRE2']
    med = np.nanmedian(diff)

    # -------------------------------------------------------------------------
    # Skip low flux files (but don't permanently reject - might want later)
    # -------------------------------------------------------------------------
    if med < MIN_FLUX_THRESHOLD:
        print(f"  -> Skipped (flux too low: {med:.1f})")
        continue

    # Normalize the difference by median to get relative variations
    diff = diff / med
    
    # Compute RMS as a stability metric
    rms = np.nanstd(diff)

    # -------------------------------------------------------------------------
    # KS test: first half vs second half of the sky-subtracted posemeter
    # A low p-value means the two halves are unlikely to be drawn from the
    # same distribution, indicating a change mid-exposure (e.g. clouds, guiding).
    # -------------------------------------------------------------------------
    mid = len(diff) // 2
    if mid >= 2:
        ks_stat, ks_pval = stats.ks_2samp(diff[:mid], diff[mid:])
    else:
        ks_stat = 0.0  # not enough points to test

    # Get observation time from index (no extra FITS header read needed)
    mjd = index[os.path.basename(file)]['mjd']

    print(f"  -> RMS: {rms:.2e}, Flux: {med:.1f}, KS D={ks_stat:.3f}, MJD: {mjd:.5f}")

    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    rms_vec.append(rms)
    med_vec.append(med)
    mjd_vec.append(mjd)
    file_vec.append(file)
    ks_stat_vec.append(ks_stat)

    # -------------------------------------------------------------------------
    # Store detailed data for high-RMS files (plotted later)
    # -------------------------------------------------------------------------
    if rms > HIGH_RMS_THRESHOLD:
        high_rms_data.append({
            'file': file,
            'time': tbl['TIME'].copy(),
            'fibre1': tbl['FIBRE1'].copy(),
            'fibre2': tbl['FIBRE2'].copy(),
            'diff': diff.copy(),
            'object': object_name,
            'rms': rms
        })
        print(f"  -> FLAGGED as high RMS!")

print("-" * 60)

# =============================================================================
# CONVERT TO NUMPY ARRAYS
# =============================================================================

rms_vec = np.array(rms_vec)
med_vec = np.array(med_vec)
mjd_vec = np.array(mjd_vec)
file_vec = np.array(file_vec)
ks_stat_vec = np.array(ks_stat_vec)

print(f"\nAnalyzed {len(rms_vec)} science observations")

if len(rms_vec) == 0:
    print("No observations to plot. Check object filter and reject list.")
    exit(0)

# =============================================================================
# PRINT HIGH-RMS FILES
# =============================================================================

high_rms_mask = rms_vec > HIGH_RMS_THRESHOLD
if np.any(high_rms_mask):
    print(f"\n{'='*60}")
    print(f"WARNING: {np.sum(high_rms_mask)} files with RMS > {HIGH_RMS_THRESHOLD}:")
    print(f"{'='*60}")
    for f, r in zip(file_vec[high_rms_mask], rms_vec[high_rms_mask]):
        print(f"  {f}: RMS = {r:.3f}")

# =============================================================================
# CREATE OUTPUT DIRECTORY FOR PLOTS
# =============================================================================

os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# PLOT 1: RMS VS DATE (SUMMARY PLOT)
# =============================================================================
# This plot shows the stability (RMS) of all observations over time,
# with points color-coded by flux level (MED)

print(f"\nGenerating summary plot...")

# Clamp flux values to 0-100 for colorbar (avoids outliers dominating scale)
med_clamped = np.clip(med_vec, 0, 100)

# Convert MJD to datetime for readable x-axis labels
dates = Time(mjd_vec, format='mjd').to_datetime()

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Scatter plot with color-coded flux
scatter = ax.scatter(dates, rms_vec, c=med_clamped, cmap='viridis', 
                     vmin=0, vmax=100, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Median Flux (clamped 0-100)')

# Add horizontal line at threshold
ax.axhline(y=HIGH_RMS_THRESHOLD, color='red', linestyle='--', linewidth=2, 
           label=f'Threshold (RMS={HIGH_RMS_THRESHOLD})')

# Labels and formatting
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('RMS of (FIBRE1 - FIBRE2) / median', fontsize=12)
ax.set_title('NIRPS Posemeter Stability Analysis', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45, ha='right')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Set y-axis to log scale if range is large
if rms_vec.max() / rms_vec.min() > 10:
    ax.set_yscale('log')

plt.tight_layout()

# Save plot
summary_plot_path = os.path.join(PLOT_DIR, 'posemeter_summary.png')
plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {summary_plot_path}")
plt.show()

# =============================================================================
# PLOT 2: HIGH-RMS OFFENDERS (DETAILED PLOT)
# =============================================================================
# This plot shows the detailed time series for all observations flagged
# as having high RMS, overplotted with different colors for comparison

if high_rms_data:
    print(f"\nGenerating offenders plot ({len(high_rms_data)} files)...")
    
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
    
    # Use distinct colors for each observation
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(high_rms_data), 10)))
    
    for i, (data, color) in enumerate(zip(high_rms_data, colors)):
        # Convert time to minutes since start of exposure
        time_min = (data['time'] - data['time'][0]) * 24 * 60  # days -> minutes
        
        # Create label with object name and RMS value
        label = f"{data['object']} (RMS: {data['rms']:.3f})"
        
        # Plot raw fiber signals (top panel)
        ax[0].plot(time_min, data['fibre1'], color=color, alpha=0.7, linewidth=1)
        ax[0].plot(time_min, data['fibre2'], color=color, alpha=0.7, 
                   linewidth=1, linestyle='--')
        
        # Plot normalized difference (bottom panel)
        ax[1].plot(time_min, data['diff'], color=color, alpha=0.7, 
                   linewidth=1, label=label)
    
    # Top panel formatting
    ax[0].set_ylabel('Raw Flux (counts)', fontsize=11)
    ax[0].set_title('High RMS Observations - Fiber Signals\n(solid = FIBRE1, dashed = FIBRE2)', 
                    fontsize=12, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    
    # Bottom panel formatting
    ax[1].set_ylabel('(FIBRE1 - FIBRE2) / median', fontsize=11)
    ax[1].set_xlabel('Time since exposure start (minutes)', fontsize=11)
    #ax[1].legend(loc='best', fontsize='small', ncol=2)
    ax[1].grid(True, alpha=0.3)
    
    # Add horizontal reference lines at ±threshold
    ax[1].axhline(y=1 + HIGH_RMS_THRESHOLD, color='red', linestyle=':', alpha=0.5)
    ax[1].axhline(y=1 - HIGH_RMS_THRESHOLD, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    offenders_plot_path = os.path.join(PLOT_DIR, 'posemeter_offenders.png')
    plt.savefig(offenders_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {offenders_plot_path}")
    plt.show()
else:
    print("\nNo high-RMS observations found - no offenders plot generated")

# =============================================================================
# PLOT 3: KS D-STATISTIC VS DATE
# =============================================================================
# The KS D-statistic is the maximum absolute difference between the CDFs of
# the first and second halves of the posemeter time series.  It ranges from
# 0 (identical distributions) to 1 (fully separated distributions) and is
# independent of sample size, making it meaningful with only a few tens of
# points per observation.

print(f"\nGenerating KS D-statistic plot...")

# Flag threshold: log10(1-D) < -1  ->  D > 0.9
ks_flag_threshold_log = -1.0  # log10(1-D)

# log10(1-D): 0 means D=0 (identical halves), strongly negative means D→1
log_one_minus_d = np.log10(np.clip(1.0 - ks_stat_vec, 1e-10, 1.0))

fig, ax = plt.subplots(figsize=(12, 6))

# Color-code by RMS so the two metrics can be compared at a glance
rms_clamped = np.clip(rms_vec, 0, HIGH_RMS_THRESHOLD * 3)
scatter = ax.scatter(dates, log_one_minus_d, c=rms_clamped,
                     cmap='hot_r',
                     vmin=0, vmax=HIGH_RMS_THRESHOLD * 3,
                     s=60, alpha=0.8, edgecolors='k', linewidths=0.5,
                     zorder=3)
cbar = plt.colorbar(scatter, ax=ax,
                    label=f'RMS of (FIBRE1-FIBRE2)/median (clamped at {HIGH_RMS_THRESHOLD*3:.2f})')

# Reference line at flag threshold
ax.axhline(y=ks_flag_threshold_log, color='red', linestyle='--', linewidth=2,
           label=f'log$_{{10}}$(1-D) = {ks_flag_threshold_log}  (D > 0.9)')

# Annotate the most discrepant observations
n_label = min(5, np.sum(log_one_minus_d < ks_flag_threshold_log))
if n_label > 0:
    worst_idx = np.argsort(log_one_minus_d)[:n_label]
    for idx in worst_idx:
        ax.annotate(os.path.basename(file_vec[idx]).replace('_posemeter.fits', ''),
                    xy=(dates[idx], log_one_minus_d[idx]),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=6, color='darkred', alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('log$_{10}$(1 - KS D)  [first half vs second half]', fontsize=12)
ax.set_title('NIRPS Posemeter — KS Test: Intra-Exposure Stability', fontsize=14,
             fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45, ha='right')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left')

plt.tight_layout()

ks_plot_path = os.path.join(PLOT_DIR, 'posemeter_ks_test.png')
plt.savefig(ks_plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {ks_plot_path}")
plt.show()

# Count how many observations are flagged by the KS test
n_ks_flagged = int(np.sum(log_one_minus_d < ks_flag_threshold_log))
print(f"Observations with log10(1-D) < {ks_flag_threshold_log}: {n_ks_flagged} / {len(ks_stat_vec)}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Total files analyzed:     {len(rms_vec)}")
print(f"High RMS (>{HIGH_RMS_THRESHOLD}):       {np.sum(high_rms_mask)}")
print(f"Median RMS:               {np.median(rms_vec):.4f}")
print(f"Mean RMS:                 {np.mean(rms_vec):.4f}")
print(f"Min/Max RMS:              {rms_vec.min():.4f} / {rms_vec.max():.4f}")
print(f"KS flagged (log10(1-D)<{ks_flag_threshold_log}): {n_ks_flagged}")
print(f"Median KS D-statistic:    {np.median(ks_stat_vec):.4f}")
print(f"Date range:               {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"{'='*60}")
print(f"\nPlots saved to: {PLOT_DIR}/")
print("Done!")
# 