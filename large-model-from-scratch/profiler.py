from collections import defaultdict

import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table

## There's a lot of code here and can be definitely use
## use some clean up and refactoring...

class CustomProfiler:
    # TODO: some numbers have been hardcoded
    # and should be passed as args to __init__
    def __init__(self, model):
        self.model = model
        self.profiler = None
        self.profile_start_step = 0
        self.model_size_bytes = self._get_model_size_bytes()

        self.profile_every_n_steps = 100
        self.warmup_steps = 2
        self.profile_duration = 10

    def _get_model_size_bytes(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size_bytes = next(self.model.parameters()).element_size()
        return total_params * param_size_bytes

    def step_start(self, current_step: int, rank: int):
        if rank == 0 and current_step > 0 and current_step % self.profile_every_n_steps == 0:
            self.profile_start_step = current_step
            schedule = torch.profiler.schedule(
                wait=self.warmup_steps, warmup=1, active=self.profile_duration
            )
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=self._on_trace_ready,
                record_shapes=False,
                with_stack=False,
            )
            print(
                f"\n--- [Rank {rank}] Profiler :: Starting profiling at step {current_step} for 5 steps ---"
            )

    def step_end(self, current_step: int, rank: int):
        if self.profiler:
            self.profiler.step()
            if current_step >= self.profile_start_step + self.warmup_steps + self.profile_duration:
                self.profiler.stop()
                if rank == 0:
                    print(
                        f"--- [Rank {rank}] Profiler: Profiling finished at step {current_step}. ---"
                    )
                self.profiler = None

    def _on_trace_ready(self, prof: torch.profiler.profile):
        raise NotImplementedError


class IBProfiler(CustomProfiler):
    def _on_trace_ready(self, prof: torch.profiler.profile):
        if dist.get_rank() != 0:
            return

        comm_ops = defaultdict(lambda: {"count": 0, "total_duration_us": 0.0})
        total_comm_time_us = 0.0

        events = prof.key_averages()
        for event in events:
            if "nccl" in event.key:
                comm_time = (
                    event.device_time * event.count
                )  # cuda_time is deprecated. use device_time instead
                comm_ops[event.key]["count"] += event.count
                comm_ops[event.key]["total_duration_us"] += comm_time
                total_comm_time_us += comm_time

        print("\n--- InfiniBand Performance Report ---")
        sorted_ops = sorted(
            comm_ops.items(), key=lambda item: item[1]["total_duration_us"], reverse=True
        )
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Communication Op", justify="left", overflow="fold", max_width=60)
        table.add_column("Count", justify="right")
        table.add_column("Total Time (ms)", justify="right")
        table.add_column("Avg Time (us)", justify="right")

        for op, data in sorted_ops:
            total_ms = data["total_duration_us"] / 1000
            avg_us = data["total_duration_us"] / data["count"]
            table.add_row(op, str(data["count"]), f"{total_ms:.3f}", f"{avg_us:.2f}")

        console.print(table)

        num_gpus = dist.get_world_size()
        if total_comm_time_us > 0 and self.model_size_bytes > 0 and num_gpus > 1:
            time_per_step_s = (total_comm_time_us / 1_000_000.0) / self.profile_duration
            effective_bw_gbps = (
                (2 * (num_gpus - 1) / num_gpus * self.model_size_bytes)
                / (1024**3)
                / time_per_step_s
            )
            print(f"Total Communication Time (per step): {time_per_step_s * 1000:.2f} ms")
            print(f"Effective All-Reduce Bandwidth: {effective_bw_gbps:.2f} GB/s")

        print("--- End of Report ---\n")


class CombinedProfiler(CustomProfiler):
    def _on_trace_ready(self, prof: torch.profiler.profile):
        if dist.get_rank() != 0:
            return

        event_list = list(prof.key_averages(group_by_input_shape=True))

        # --- Part 1: InfiniBand Performance Analysis ---
        comm_ops = defaultdict(lambda: {"count": 0, "total_duration_us": 0.0})
        total_comm_time_us = 0.0

        for event in event_list:
            # NCCL kernels are used for inter-GPU communication.
            if "nccl" in event.key:
                comm_time = event.device_time_total
                comm_ops[event.key]["count"] += event.count
                comm_ops[event.key]["total_duration_us"] += comm_time
                total_comm_time_us += comm_time

        console = Console()
        print("\n--- InfiniBand Performance Report ---")
        ib_table = Table(show_header=True, header_style="bold magenta", expand=True)
        ib_table.add_column("Communication Op", justify="left", overflow="fold", max_width=60)
        ib_table.add_column("Count", justify="right")
        ib_table.add_column("Total Time (ms)", justify="right")
        ib_table.add_column("Avg Time (us)", justify="right")

        sorted_ops = sorted(
            comm_ops.items(), key=lambda item: item[1]["total_duration_us"], reverse=True
        )

        for op, data in sorted_ops:
            if data["count"] > 0:
                total_ms = data["total_duration_us"] / 1000
                avg_us = data["total_duration_us"] / data["count"]
                ib_table.add_row(op, str(data["count"]), f"{total_ms:.3f}", f"{avg_us:.2f}")

        console.print(ib_table)

        num_gpus = dist.get_world_size()
        if total_comm_time_us > 0 and self.model_size_bytes > 0 and num_gpus > 1:
            time_per_step_s = (total_comm_time_us / 1_000_000.0) / self.profile_duration
            effective_bw_gbps = (
                (2 * (num_gpus - 1) / num_gpus * self.model_size_bytes)
                / (1024**3)
                / time_per_step_s
            )
            print(f"Total Communication Time (per step): {time_per_step_s * 1000:.2f} ms")
            print(f"Effective All-Reduce Bandwidth: {effective_bw_gbps:.2f} GB/s")

        print("--- End of InfiniBand Report ---")

        # --- Part 2: CUDA Task Performance Analysis ---
        total_gpu_time_us = sum(event.device_time_total for event in event_list)

        print("\n--- CUDA Task Performance Report ---")
        cuda_table = Table(show_header=True, header_style="bold cyan", expand=True)
        cuda_table.add_column("CUDA Kernel", justify="left", overflow="fold", max_width=60)
        cuda_table.add_column("Count", justify="right")
        cuda_table.add_column("Total Time (ms)", justify="right")
        cuda_table.add_column("Percentage", justify="right")

        sorted_events = sorted(event_list, key=lambda e: e.device_time_total, reverse=True)

        for event in sorted_events:
            total_ms = event.device_time_total / 1000
            percentage = (
                (event.device_time_total / total_gpu_time_us) * 100 if total_gpu_time_us > 0 else 0
            )
            if total_ms > 0.001:  # Only show significant events
                cuda_table.add_row(
                    event.key, f"{event.count}", f"{total_ms:.3f}", f"{percentage:.2f}%"
                )

        console.print(cuda_table)
        print("--- End of CUDA Report ---\n")


class CUDATaskProfiler(CustomProfiler):
    def _on_trace_ready(self, prof: torch.profiler.profile):
        if dist.get_rank() != 0:
            return

        # group by input shapes to get a more detailed view
        events = prof.key_averages(group_by_input_shape=True)
        total_gpu_time_us = sum(event.device_time_total for event in events)

        print("\n--- CUDA Task Performance Report ---")
        console = Console()
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("CUDA Kernel", justify="left", overflow="fold", max_width=60)
        table.add_column("Count", justify="right")
        table.add_column("Total Time (ms)", justify="right")
        table.add_column("Percentage", justify="right")

        sorted_events = sorted(events, key=lambda e: e.device_time_total, reverse=True)

        for event in sorted_events:
            total_ms = event.device_time_total / 1000
            percentage = (
                (event.device_time_total / total_gpu_time_us) * 100 if total_gpu_time_us > 0 else 0
            )
            table.add_row(event.key, f"{event.count}", f"{total_ms:.3f}", f"{percentage:.2f}%")

        console.print(table)
        print("--- End of Report ---\n")
