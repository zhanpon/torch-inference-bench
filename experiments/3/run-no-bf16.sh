set -x

# fp32 resnet benchmarks
python torchvision_bench.py --channels-last --model-name=resnet50
python torchvision_bench.py --channels-last --model-name=resnet101
python torchvision_bench.py --channels-last --model-name=resnet152

# fp32 resnet profiling
python torchvision_bench.py --profile --channels-last --model-name=resnet50
python torchvision_bench.py --profile --channels-last --model-name=resnet101
python torchvision_bench.py --profile --channels-last --model-name=resnet152

# CPU info
lscpu


