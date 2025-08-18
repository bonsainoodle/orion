# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Orion is a Fully Homomorphic Encryption (FHE) framework for deep learning that enables privacy-preserving neural network inference. The framework consists of Python frontend components with optimized backends (primarily Lattigo in Go).

## Installation & Setup

This project requires Go 1.22+ for the Lattigo backend. Install dependencies:
```bash
pip install -e .
```

The build process automatically compiles the Go shared library using `tools/build_lattigo.py`.

## Core Architecture

### Backend System
- **Primary Backend**: Lattigo (Go-based FHE library)
- **Backend Interfaces**: `orion/backend/` contains Python bindings for different FHE backends
- **Shared Library**: Go code is compiled to platform-specific shared libraries (`.so`, `.dylib`, `.dll`)

### Core Components
- **Scheme (`orion/core/orion.py`)**: Main orchestrator that manages FHE operations and workflows
- **Tracer (`orion/core/tracer.py`)**: Uses PyTorch FX to trace models and collect FHE statistics  
- **Network DAG (`orion/core/network_dag.py`)**: Builds computation graph for optimization
- **Auto Bootstrap (`orion/core/auto_bootstrap.py`)**: Automatic bootstrap placement optimization

### Neural Network Modules
- **Base Module (`orion/nn/module.py`)**: Base class for FHE-compatible layers
- **Linear Transforms (`orion/nn/linear.py`)**: Convolutions, fully-connected layers
- **Activations (`orion/nn/activation.py`)**: Polynomial approximations (ReLU, SiLU, etc.)

## Key Workflows

### Model Preparation
1. **Fitting**: `scheme.fit(model, input_data)` - Traces model, collects statistics, fits polynomial activations
2. **Compilation**: `scheme.compile(model)` - Builds DAG, fuses modules, packs matrices, places bootstraps

### Configuration
- Models configured via YAML files in `configs/` directory
- Configuration specifies FHE parameters, backend choice, optimization settings

## Development Commands

### Running Examples
```bash
cd examples/
python3 run_lola.py          # LOLA architecture example
python3 run_mlp.py           # MLP example  
python3 run_resnet18_relu_cifar10.py  # ResNet example
```

### Testing
```bash
python -m pytest tests/      # Run all tests
python -m pytest tests/models/test_mlp.py  # Run specific test
```

### Building Backend
The Go backend is built automatically during installation, but can be rebuilt manually:
```bash
python tools/build_lattigo.py
```

## Important Implementation Details

### FHE-Specific Constraints
- Batch dimensions must be consistent across the network
- Pooling operations require equal stride in all directions  
- BatchNorm layers cannot have multiple parent nodes (prevents fusion)
- Input/output ranges are tracked for polynomial fitting

### Level Management
- Each operation consumes "levels" (noise budget in FHE)
- Bootstrap operations refresh noise budget but are expensive
- Automatic bootstrap placement optimizes performance vs. accuracy trade-offs

### Packing Strategy  
- Tensors are packed into SIMD slots for efficient FHE operations
- "Gap" parameters control data layout in ciphertext slots
- Matrix diagonals are pre-computed for linear transformations

## File Structure Notes

- `models/` and `orion/models/`: Model definitions (some duplication for compatibility)
- `orion/backend/lattigo/`: Go implementation with Python bindings
- `orion/backend/python/`: Pure Python FHE implementation (reference/testing)
- `examples/`: Runnable examples for different model architectures
- `configs/`: YAML configuration files for FHE parameters