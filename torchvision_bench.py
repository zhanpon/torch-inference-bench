import click
import torch
import torch.utils.benchmark as benchmark
import torchvision
import torch.profiler as profiler


@click.command()
@click.option("--profile", is_flag=True)
@click.option("--model-name", default="resnet50")
@click.option("--iterations", type=int, default=100)
@click.option("--batch-size", type=int, default=1)
@click.option("--num-threads", type=int, default=1)
@click.option("--bf16", "autocast_bf16", is_flag=True)
@click.option("--quantize", is_flag=True)
@click.option("--channels-last", is_flag=True)
@click.option("--jit", is_flag=True)
@click.option("--compile", "with_compile", is_flag=True)
def main(
    profile: bool,
    model_name: str,
    iterations: int,
    batch_size: int,
    num_threads: int,
    autocast_bf16: bool,
    quantize: bool,
    channels_last: bool,
    jit: bool,
    with_compile: bool,
):
    x = torch.randn((batch_size, 3, 224, 224))

    if quantize:
        factory = getattr(torchvision.models.quantization, model_name)
        m = factory(quantize=True)
    else:
        factory = getattr(torchvision.models, model_name)
        m = factory()
    m.eval()

    if channels_last:
        x = x.to(memory_format=torch.channels_last)
        m = m.to(memory_format=torch.channels_last)

    # See https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    if jit:
        m = torch.jit.optimize_for_inference(torch.jit.script(m))

    if with_compile:
        m = torch.compile(m)

    @torch.autocast("cpu", torch.bfloat16, enabled=autocast_bf16)
    @torch.inference_mode()
    def do_infer():
        return m(x)

    if profile:
        torch.set_num_threads(num_threads)
        for _ in range(3):
            do_infer()

        with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
            with profiler.record_function("do_infer"):
                do_infer()

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    else:
        t = benchmark.Timer(
            stmt="do_infer()",
            globals={"do_infer": do_infer},
            num_threads=num_threads,
        )

        print(t.timeit(iterations))


if __name__ == "__main__":
    main()
