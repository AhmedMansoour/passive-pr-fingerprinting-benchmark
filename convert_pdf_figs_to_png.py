import fitz
from pathlib import Path

fig_dir = Path("paper_outputs/figures")

pdf_files = [
    "model_comparison_comprehensive.pdf",
    "methods_bxp_statistical.pdf",
    "transformers_arch_pointwise_errors_with_box.pdf",
    "ap_density_sensitivity.pdf",
    "ann_architectures_scaled.pdf",
    "arch3_arch4_xy_meters_transformer.pdf",
]

for name in pdf_files:
    pdf_path = fig_dir / name
    if not pdf_path.exists():
        print(f"[SKIP] Missing: {pdf_path}")
        continue

    out_path = fig_dir / f"{pdf_path.stem}.png"
    doc = fitz.open(pdf_path)
    page = doc[0]

    # 3x scaling gives good README quality
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    pix.save(out_path)

    print(f"[OK] {pdf_path} -> {out_path}")