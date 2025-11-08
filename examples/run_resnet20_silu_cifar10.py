import time
import math
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_cifar_datasets,
    get_tiny_datasets,
    mae,
    train_on_cifar
)

# Set seed for reproducibility
torch.manual_seed(42)

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Setup logging to both file and stdout
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"run_resnet20_silu_cifar10_{timestamp}.log"

class TeeOutput:
    """Write to both stdout and a file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Redirect stdout to both terminal and file
tee = TeeOutput(log_file)
sys.stdout = tee

print(f"Logging to: {log_file}")
print(f"Started at: {datetime.now().isoformat()}")
print("="*80)

# Initialize timing storage
layer_timings = []
timing_hooks = []

def create_timing_hook(layer_name):
    """Create a forward hook that times layer execution in FHE mode"""
    def hook(module, input, output):
        if hasattr(module, 'he_mode') and module.he_mode:
            # Timing is already done by the @timer decorator
            # We just need to record the layer name for reference
            pass
    return hook

def register_timing_hooks(model):
    """Register forward hooks on all Orion modules to capture layer names"""
    layer_count = {}
    for name, module in model.named_modules():
        if hasattr(module, 'he_mode'):
            # Count module types for unique naming
            module_type = type(module).__name__
            layer_count[module_type] = layer_count.get(module_type, 0) + 1
            layer_name = f"{module_type}_{layer_count[module_type]}"

            # Store the name on the module for the timer decorator
            if not hasattr(module, 'name'):
                module.name = layer_name

            hook = module.register_forward_hook(create_timing_hook(layer_name))
            timing_hooks.append(hook)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
net = models.ResNet20SiLU("cifar10")

# Register timing hooks
register_timing_hooks(net)

# Get a test batch to pass through our network
inp, _ = next(iter(testloader))

# Run cleartext inference
print("Running cleartext inference...")
net.eval()
start_clear = time.time()
out_clear = net(inp)
end_clear = time.time()
clear_time = end_clear - start_clear

# Prepare for FHE inference
print("Fitting model...")
fit_start = time.time()
orion.fit(net, inp)
fit_time = time.time() - fit_start

print("Compiling model...")
compile_start = time.time()
input_level = orion.compile(net)
compile_time = time.time() - compile_start

# Encode and encrypt the input vector
print("Encoding and encrypting input...")
encode_start = time.time()
vec_ptxt = orion.encode(inp, input_level)
encode_time = time.time() - encode_start

encrypt_start = time.time()
vec_ctxt = orion.encrypt(vec_ptxt)
encrypt_time = time.time() - encrypt_start

net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start_fhe = time.time()
out_ctxt = net(vec_ctxt)
end_fhe = time.time()
fhe_inference_time = end_fhe - start_fhe

# Decrypt and decode
print("Decrypting and decoding output...")
decrypt_start = time.time()
out_ptxt = out_ctxt.decrypt()
decrypt_time = time.time() - decrypt_start

decode_start = time.time()
out_fhe = out_ptxt.decode()
decode_time = time.time() - decode_start

# Compare the cleartext and FHE results
print()
print("Clear output:", out_clear)
print("FHE output:", out_fhe)

dist = mae(out_clear, out_fhe)
precision = -math.log2(dist) if dist > 0 else float('inf')

print(f"\nMAE: {dist:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Runtime: {fhe_inference_time:.4f} secs.\n")

# Prepare comprehensive results
total_time = (fit_time + compile_time + encode_time + encrypt_time +
              fhe_inference_time + decrypt_time + decode_time)

results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "model": "ResNet20-SiLU",
        "dataset": "CIFAR-10",
        "framework": "Orion",
        "config_file": "../configs/resnet.yml",
        "input_shape": list(inp.shape),
        "batch_size": 1,
        "seed": 42
    },
    "timing": {
        "cleartext_inference_sec": round(clear_time, 4),
        "fit_sec": round(fit_time, 4),
        "compile_sec": round(compile_time, 4),
        "encode_sec": round(encode_time, 4),
        "encrypt_sec": round(encrypt_time, 4),
        "fhe_inference_sec": round(fhe_inference_time, 4),
        "decrypt_sec": round(decrypt_time, 4),
        "decode_sec": round(decode_time, 4),
        "total_fhe_pipeline_sec": round(total_time, 4),
        "overhead_factor": round(fhe_inference_time / clear_time, 2) if clear_time > 0 else None
    },
    "accuracy": {
        "mae": round(dist, 6),
        "precision_bits": round(precision, 4) if precision != float('inf') else "inf",
        "cleartext_output_sample": out_clear[0][:5].tolist() if len(out_clear[0]) >= 5 else out_clear[0].tolist(),
        "fhe_output_sample": out_fhe[0][:5].tolist() if len(out_fhe[0]) >= 5 else out_fhe[0].tolist()
    },
    "system_info": {
        "python_version": f"{torch.version.__version__}",
        "torch_version": torch.__version__,
        "device": "cpu"
    },
    "notes": [
        "Debug mode enabled in config - per-layer timing printed to stdout",
        "Per-layer timing details available in debug output",
        "Using random data (no actual dataset download)",
        "FHE scheme: CKKS with auto-bootstrap",
        "All layers encrypted (full network)"
    ]
}

# Save results to JSON
output_file = log_dir / f"run_resnet20_silu_cifar10_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print(f"Results saved to: {output_file}")
print(f"Log file saved to: {log_file}")
print(f"Total FHE pipeline time: {total_time:.2f} seconds")
print(f"FHE overhead factor: {results['timing']['overhead_factor']}x")
print(f"Completed at: {datetime.now().isoformat()}")
print("="*80)

# Clean up hooks
for hook in timing_hooks:
    hook.remove()

# Restore stdout and close log file
sys.stdout = tee.terminal
tee.close()
