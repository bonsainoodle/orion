import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_cifar_datasets,
    get_tiny_datasets,
    mae,
    train_on_cifar
)
from benchmark_logger import BenchmarkLogger

# Initialize logger
logger = BenchmarkLogger("resnet20_relu_cifar10")
logger.set_config("../configs/resnet.yml")
logger.set_model_info(
    name="ResNet-20",
    architecture="ResNet20",
    dataset="CIFAR-10",
    batch_size=1,
    seed=42
)

try:
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize the Orion scheme, model, and data
    logger.start_phase("initialization")
    scheme = orion.init_scheme("../configs/resnet.yml")
    logger.end_phase("initialization")

    logger.start_phase("data_loading")
    trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
    net = models.ResNet20("cifar10")
    inp, _ = next(iter(testloader))
    logger.end_phase("data_loading")

    # Train model (optional)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_on_cifar(net, data_dir="../data", epochs=1, device=device)

    # Run cleartext inference
    logger.start_phase("cleartext_inference")
    net.eval()
    out_clear = net(inp)
    logger.end_phase("cleartext_inference")

    # Prepare for FHE inference.
    # Some polynomial activation functions require knowing the range of possible
    # input values. We'll estimate these ranges using training set statistics,
    # adjusted to be wider by a tolerance factor (= margin).
    logger.start_phase("fit")
    orion.fit(net, inp)
    logger.end_phase("fit")

    logger.start_phase("compile")
    input_level = orion.compile(net)
    logger.end_phase("compile", input_level=input_level)

    # Encode and encrypt the input vector
    logger.start_phase("encode")
    vec_ptxt = orion.encode(inp, input_level)
    logger.end_phase("encode")

    logger.start_phase("encrypt")
    vec_ctxt = orion.encrypt(vec_ptxt)
    logger.end_phase("encrypt")

    net.he()  # Switch to FHE mode

    # Run FHE inference
    logger.start_phase("fhe_inference")
    out_ctxt = net(vec_ctxt)
    logger.end_phase("fhe_inference")

    # Get the FHE results and decrypt + decode.
    logger.start_phase("decrypt")
    out_ptxt = out_ctxt.decrypt()
    logger.end_phase("decrypt")

    logger.start_phase("decode")
    out_fhe = out_ptxt.decode()
    logger.end_phase("decode")

    # Compare the cleartext and FHE results.
    print()
    print("Cleartext output sample:")
    print(out_clear)
    print("\nFHE output sample:")
    print(out_fhe)

    dist = mae(out_clear, out_fhe)
    precision = -math.log2(dist)

    print(f"\nMAE: {dist:.6f}")
    print(f"Precision: {precision:.4f} bits")

    # Log results
    logger.set_results(dist, precision, out_clear, out_fhe)
    logger.finalize(status="success")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    logger.finalize(status="failed", error=str(e))
    raise