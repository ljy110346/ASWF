# Data Format

ASWF expects row-wise paired infrared and Raman spectra. Each dataset is stored in its own directory under `data/`.

## Directory Layout

```text
data/
└── example_disease/
    ├── ir_healthy.xlsx
    ├── raman_healthy.xlsx
    ├── ir_disease.xlsx
    ├── raman_disease.xlsx
    ├── Wavenumber-ir.xlsx        # optional
    └── Wavenumber-raman.xlsx     # optional
```

## Matrix Convention

Each Excel file should contain a numeric matrix:

- rows: samples
- columns: spectral variables

The healthy infrared and healthy Raman files must have the same number of rows. The disease infrared and disease Raman files must also have the same number of rows. ASWF pairs the two modalities row by row within each class.

## Wavenumber Axes

Optional wavenumber files should contain one column or one row with the spectral axis values. If an axis file is missing or has a length mismatch, the behavior is controlled by `data.axis_mismatch_policy`:

- `warn`: continue and use feature indices where needed
- `error`: stop with an exception

## Configuration Example

```yaml
data:
  root: data
  diseases:
    - example_disease
  file_mappings:
    example_disease:
      ir_normal: ir_healthy.xlsx
      raman_normal: raman_healthy.xlsx
      ir_abnormal: ir_disease.xlsx
      raman_abnormal: raman_disease.xlsx
      wavenumber_ir: Wavenumber-ir.xlsx
      wavenumber_raman: Wavenumber-raman.xlsx
```

The terms `normal` and `abnormal` are internal class names used by the loader. They correspond to healthy/control and disease/positive classes, respectively.
