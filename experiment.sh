set -eux

# resnet50 benchmarks
python torchvision_bench.py
python torchvision_bench.py --channels-last
python torchvision_bench.py --bf16
python torchvision_bench.py --bf16 --channels-last
python torchvision_bench.py --quantize
python torchvision_bench.py --quantize --channels-last

# convnext_tiny benchmarks
python torchvision_bench.py --model-name convnext_tiny
python torchvision_bench.py --model-name convnext_tiny --channels-last
python torchvision_bench.py --model-name convnext_tiny --bf16
python torchvision_bench.py --model-name convnext_tiny --bf16 --channels-last

# resnet50 profiling
python torchvision_bench.py --profile
python torchvision_bench.py --profile --channels-last
python torchvision_bench.py --profile --bf16
python torchvision_bench.py --profile --bf16 --channels-last
python torchvision_bench.py --profile --quantize
python torchvision_bench.py --profile --quantize --channels-last

# convnext_tiny profiling
python torchvision_bench.py --profile --model-name convnext_tiny
python torchvision_bench.py --profile --model-name convnext_tiny --channels-last
python torchvision_bench.py --profile --model-name convnext_tiny --bf16
python torchvision_bench.py --profile --model-name convnext_tiny --bf16 --channels-last

# CPU info
lscpu
