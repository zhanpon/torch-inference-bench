set -x

# fp32 resnet benchmarks
python torchvision_bench.py --channels-last --model-name=resnet50
python torchvision_bench.py --channels-last --model-name=resnet101
python torchvision_bench.py --channels-last --model-name=resnet152

# bf16 resnet benchmarks
python torchvision_bench.py --channels-last --model-name=resnet50 --bf16
python torchvision_bench.py --channels-last --model-name=resnet101 --bf16
python torchvision_bench.py --channels-last --model-name=resnet152 --bf16

# fp32 resnet profiling
python torchvision_bench.py --profile --channels-last --model-name=resnet50
python torchvision_bench.py --profile --channels-last --model-name=resnet101
python torchvision_bench.py --profile --channels-last --model-name=resnet152

# bf16 resnet profiling
python torchvision_bench.py --profile --channels-last --model-name=resnet50 --bf16
python torchvision_bench.py --profile --channels-last --model-name=resnet101 --bf16
python torchvision_bench.py --profile --channels-last --model-name=resnet152 --bf16

# CPU info
lscpu


