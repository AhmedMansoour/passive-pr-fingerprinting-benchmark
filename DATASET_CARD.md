# Dataset Card

## Dataset name

Passive Wi-Fi Probe Request Fingerprinting Localization Benchmark

## Purpose

The dataset supports reproducible benchmarking of passive Wi-Fi probe-request fingerprinting methods for indoor localization in one controlled indoor testbed.

## Data composition

- 36 training/reference locations.
- 9 held-out test locations.
- 6 monitor-node/AP RSSI channels.
- 9,373 raw training frames.
- 2,322 raw test frames.
- Derived mean-RSSI fingerprint matrices for direct model evaluation.

## Collection setting

Probe-request frames were passively captured from Wi-Fi transmissions. The benchmark uses RSSI values associated with six monitor nodes/AP identifiers and fixed ground-truth coordinates.

## Preprocessing

For the benchmark scripts, raw frames are aggregated by location and AP using mean RSSI. Missing AP observations are filled with `-100` dBm. Coordinates are stored in millimeters and converted to meters only when reporting localization errors.

## Anonymization

The original MAC address field has been removed from the public raw data. A deterministic pseudonymous `Device_ID` field is provided only to preserve the table structure where needed. The current benchmark does not require device identity for model training or evaluation.

## Intended uses

- Reproducing the benchmark tables and figures associated with the manuscript.
- Comparing passive PR fingerprinting algorithms under a common single-environment protocol.
- Teaching or prototyping indoor localization pipelines using small passive Wi-Fi datasets.

## Out-of-scope uses

- Claims of broad cross-building, cross-floor, or cross-device generalization.
- User tracking, surveillance, or identification.
- Security-sensitive inference about devices or people.

## Limitations

The dataset is intentionally small and single-environment. It includes 9 held-out test locations; therefore, model rankings and confidence intervals should be interpreted as benchmark-level evidence within this testbed, not as universal performance estimates.
