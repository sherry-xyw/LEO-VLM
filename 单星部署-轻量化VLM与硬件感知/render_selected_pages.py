#!/usr/bin/env python3
"""Render selected PDF pages used by the reading pack."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "参考文献" / "References"
OUT_DIR = Path(__file__).resolve().parent / "论文图片摘录"
PDFTOPPM = Path(
    "/Users/sherry/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/bin/pdftoppm"
)

PAGES = [
    ("01_Sang", "Sang2026_Onboard-RSFM-Deployment-Review_RemoteSens18-2-298.pdf", 5, "pipeline"),
    ("01_Sang", "Sang2026_Onboard-RSFM-Deployment-Review_RemoteSens18-2-298.pdf", 12, "hardware"),
    ("02_Du", "Du2025_First-On-Orbit-GeoFM_arXiv2512.01181.pdf", 5, "pipeline"),
    ("02_Du", "Du2025_First-On-Orbit-GeoFM_arXiv2512.01181.pdf", 13, "resource"),
    ("03_VLMEdge", "Sharshar2025_VLM-Edge-Networks-Survey_arXiv2502.07855.pdf", 11, "distributed"),
    ("03_VLMEdge", "Sharshar2025_VLM-Edge-Networks-Survey_arXiv2502.07855.pdf", 12, "compression"),
    ("04_SpIRIT", "SpIRIT_Loris_Onboard-AI_arXiv2404.08399.pdf", 3, "architecture"),
    ("04_SpIRIT", "SpIRIT_Loris_Onboard-AI_arXiv2404.08399.pdf", 6, "reliability"),
    ("05_FPGASurvey", "Antunes2025_FPGA-NN-Accelerators-Space-Survey_arXiv2504.16173.pdf", 5, "dataflow"),
    ("05_FPGASurvey", "Antunes2025_FPGA-NN-Accelerators-Space-Survey_arXiv2504.16173.pdf", 22, "power"),
    ("06_Shao", "Shao2025_Spaceborne-NN-Accelerator-FaultTolerance_RemoteSens17-1-69.pdf", 8, "architecture"),
    ("06_Shao", "Shao2025_Spaceborne-NN-Accelerator-FaultTolerance_RemoteSens17-1-69.pdf", 23, "fault_results"),
    ("07_MobileVLM", "Chu2023_MobileVLM_arXiv2312.16886.pdf", 5, "architecture"),
    ("07_MobileVLM", "Chu2023_MobileVLM_arXiv2312.16886.pdf", 9, "latency"),
    ("08_AWQ", "Lin2023_AWQ_arXiv2306.00978.pdf", 3, "method"),
    ("08_AWQ", "Lin2023_AWQ_arXiv2306.00978.pdf", 11, "edge_speed"),
    ("09_QViT", "Li2022_Q-ViT_arXiv2201.07703.pdf", 5, "method"),
    ("09_QViT", "Li2022_Q-ViT_arXiv2201.07703.pdf", 6, "results"),
    ("10_TinyCLIP", "Wu2023_TinyCLIP_arXiv2309.12314.pdf", 3, "distillation"),
    ("10_TinyCLIP", "Wu2023_TinyCLIP_arXiv2309.12314.pdf", 6, "results"),
    ("11_RemoteTrimmer", "Zou2024_RemoteTrimmer_arXiv2412.12603.pdf", 2, "method"),
    ("11_RemoteTrimmer", "Zou2024_RemoteTrimmer_arXiv2412.12603.pdf", 3, "results"),
    ("12_LLaVAUHD", "Xu2024_LLaVA-UHD_arXiv2403.11703.pdf", 5, "framework"),
    ("12_LLaVAUHD", "Xu2024_LLaVA-UHD_arXiv2403.11703.pdf", 8, "results"),
    ("13_KIVI", "Liu2024_KIVI_arXiv2402.02750.pdf", 5, "method"),
    ("13_KIVI", "Liu2024_KIVI_arXiv2402.02750.pdf", 8, "throughput"),
    ("14_vLLM", "Kwon2023_PagedAttention-vLLM_arXiv2309.06180.pdf", 5, "system"),
    ("14_vLLM", "Kwon2023_PagedAttention-vLLM_arXiv2309.06180.pdf", 10, "throughput"),
    ("15_MInference", "Jiang2024_MInference-1.0_arXiv2407.02490.pdf", 4, "patterns"),
    ("15_MInference", "Jiang2024_MInference-1.0_arXiv2407.02490.pdf", 7, "results"),
    ("16_LayerSkip", "Elhoushi2024_LayerSkip_arXiv2404.16710.pdf", 2, "overview"),
    ("16_LayerSkip", "Elhoushi2024_LayerSkip_arXiv2404.16710.pdf", 16, "speedup"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, pdf_name, page, label in PAGES:
        source = PDF_DIR / pdf_name
        target_stem = OUT_DIR / f"{prefix}_{label}_p{page:02d}"
        subprocess.run(
            [
                str(PDFTOPPM),
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                "150",
                "-png",
                "-singlefile",
                str(source),
                str(target_stem),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
