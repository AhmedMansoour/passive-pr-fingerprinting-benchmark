# Dataset description

This folder contains the passive Wi-Fi probe-request data used in the benchmark and the derived fingerprint matrices used by the localization scripts.

## Files

| File | Separator | Role | Locations |
|---|---|---|---:|
| `raw/df_all.csv` | `;` | Raw anonymized frames for training/reference locations | 36 |
| `raw/test_df.csv` | `;` | Raw anonymized frames for held-out evaluation locations | 9 |
| `processed/train_fingerprints.csv` | `,` | Aggregated mean-RSSI fingerprints for training/reference locations | 36 |
| `processed/test_fingerprints.csv` | `,` | Aggregated mean-RSSI fingerprints for held-out evaluation locations | 9 |

## Raw schema

| Column | Description |
|---|---|
| `Point` or `Folder` | Location identifier. `Point` is used in training; `Folder` is used in testing. |
| `AP` | Monitor node / AP identifier (`ap1`–`ap6`). |
| `Timestamp` | Frame timestamp as recorded during collection. |
| `Frequency` | Wi-Fi channel frequency in MHz. |
| `Signal` | Received signal strength in dBm. |
| `Antenna 0`–`Antenna 3` | Per-antenna RSSI fields from the capture device. |
| `Device_ID` | Deterministic pseudonymous identifier replacing the original MAC address. |

## Processed fingerprint schema

| Column | Description |
|---|---|
| `location_id` | Lowercase location identifier. |
| `X_mm`, `Y_mm` | Ground-truth coordinates in millimeters. |
| `ap1`–`ap6` | Mean RSSI for each monitor node/AP at the location. Missing AP observations are represented as `-100` dBm. |

## Coordinate system

Coordinates are expressed in millimeters relative to a fixed local origin. Reported localization errors are converted to meters by dividing Euclidean errors by 1000.

## Train/test split

The benchmark uses 36 training/reference locations and 9 spatially separated held-out test locations from the same indoor environment. The split is designed for controlled within-environment comparison and does not claim cross-environment generalization.

## Privacy note

The original hardware address field has been removed and replaced by `Device_ID`. The provided data should be used only for research and reproducibility purposes.
