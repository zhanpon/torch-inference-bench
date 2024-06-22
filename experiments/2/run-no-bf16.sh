set -eux

# resnet50 benchmarks
python torchvision_bench.py --channels-last
python torchvision_bench.py --channels-last --num-threads 2
python torchvision_bench.py --quantize --channels-last
python torchvision_bench.py --quantize --channels-last --num-threads 2

# convnext_tiny benchmarks
python torchvision_bench.py --model-name convnext_tiny --channels-last
python torchvision_bench.py --model-name convnext_tiny --channels-last --num-threads 2

# resnet50 profiling
python torchvision_bench.py --profile --channels-last
python torchvision_bench.py --profile --channels-last --num-threads 2
python torchvision_bench.py --profile --quantize --channels-last
python torchvision_bench.py --profile --quantize --channels-last --num-threads 2

# convnext_tiny profiling
python torchvision_bench.py --profile --model-name convnext_tiny --channels-last
python torchvision_bench.py --profile --model-name convnext_tiny --channels-last --num-threads 2

# CPU info
lscpu
