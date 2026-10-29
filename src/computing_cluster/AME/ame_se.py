# Copyright (c) 2026
# All rights reserved.

"""Compatibility entry point for the shared local-SPM compute SoC.

Keep the standalone AME entry on the same RVV/AME/SPMVirtDma topology as
configs/example/spm_compute_soc.py. Accelerator operands must be staged in
SPM; this entry must not reintroduce a direct accelerator-to-DDR connection.
All command-line arguments are handled by the canonical SoC configuration.
"""

from pathlib import Path
import runpy


def main():
    config = Path(__file__).resolve().parents[3] / "configs/example/spm_compute_soc.py"
    runpy.run_path(str(config), run_name="__m5_main__")


if __name__ == "__m5_main__":
    main()
