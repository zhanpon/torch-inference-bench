# torch-inference-bench
Benchmarks for PyTorch inference

## Steps to run an experiment (Ubuntu 22.04)

Set up venv:
```
sudo apt update
sudo apt install python3.10-venv
python3 -m venv --upgrade-deps .venv
source .venv/bin/activate
```

Install dependencies:
```
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.lock
```

Run benchmarks:
```
export BASH_XTRACEFD=1
bash experiments/1/run.sh | tee results.txt
```
