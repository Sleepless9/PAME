# PAME

**Performance Analysis and Bottleneck Localization for RISC-V Matrix Extensions**

PAME is a research project built upon [AME_gem5](https://github.com/OpenXiangShan/AME_gem5), aiming to develop a performance analysis and bottleneck localization framework for RISC-V Attached Matrix Extension (AME).

## Overview

PAME extends AME_gem5 with support for out-of-order (O3) CPU execution and microarchitectural modeling of matrix execution units.

The project focuses on understanding the performance behavior of matrix workloads, particularly the interactions among CPU execution, matrix computation, and memory access.

## Research Objectives

- **Performance Modeling:** Develop configurable microarchitectural models for matrix execution.
- **Performance Analysis:** Explore Roofline and hierarchical Top-down analysis methods for RISC-V AME.
- **Bottleneck Localization:** Identify performance bottlenecks across CPU, matrix execution units, and memory systems.
- **Software Optimization:** Provide performance insights to guide matrix workload optimization.

## Project Status

PAME is currently under active development.

The research builds upon O3 CPU integration and matrix microarchitecture modeling, with ongoing work toward performance profiling and bottleneck analysis.

## Acknowledgments

This project is based on [OpenXiangShan/AME_gem5](https://github.com/OpenXiangShan/AME_gem5) and the [gem5 simulator](https://www.gem5.org/).
