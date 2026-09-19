"""CUDA graph execution for fixed-size SAC batches and constant learning rates."""
import copy

import torch


class SACGraphUpdater:
    def __init__(self, agent, example_batch):
        if next(agent.parameters()).device.type != "cuda":
            raise ValueError("CUDA graph training requires a CUDA device")
        if agent.soft_target_update_rate is None:
            raise ValueError("CUDA graph training currently requires soft target updates")
        for scheduler in (agent.actor_lr_scheduler, agent.critic_lr_scheduler):
            if not isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR) or scheduler.factor != 1.0:
                raise ValueError("CUDA graph training requires constant learning rates (factor=1)")
        optimizers = [agent.critic_optimizer]
        if agent.train_actor:
            optimizers.append(agent.actor_optimizer)
        if any(not isinstance(opt, torch.optim.Adam) for opt in optimizers):
            raise ValueError("CUDA graph training currently supports Adam optimizers")
        self.agent = agent
        self.batch = {k: v.clone() for k, v in example_batch.items()}
        self.device = next(agent.parameters()).device
        # Warmup/capture must not consume real training updates or random draws.
        model_state = {k: v.clone() for k, v in agent.state_dict().items()}
        optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        for opt in optimizers:
            for group in opt.param_groups:
                group["capturable"] = True
                group["foreach"] = True
            # Adam may have initialized step counters on the CPU in eager mode.
            for state in opt.state.values():
                if "step" in state:
                    state["step"] = state["step"].to(self.device)

        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._update()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        with torch.cuda.graph(self.graph, stream=stream):
            info = self._update()
            self.metric_names = [k for k, v in info.items() if isinstance(v, torch.Tensor)]
            self.metrics = torch.stack([info[k] for k in self.metric_names])
        torch.cuda.current_stream(self.device).wait_stream(stream)
        # Preserve the tensor addresses captured by the graph when restoring Adam.
        with torch.no_grad():
            agent.load_state_dict(model_state)
            for opt, old in zip(optimizers, optimizer_states):
                for group, old_group in zip(opt.param_groups, old["param_groups"]):
                    for parameter, old_id in zip(group["params"], old_group["params"]):
                        previous = old["state"].get(old_id, {})
                        for key, value in opt.state[parameter].items():
                            if isinstance(value, torch.Tensor):
                                if key in previous:
                                    value.copy_(previous[key])
                                else:
                                    value.zero_()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, self.device)

    def _update(self):
        return self.agent.update(
            **self.batch, step=0, log_stats=True, tensor_stats=True,
            update_schedules=False,
        )

    def update(self, *, step, log_stats=True, **batch):
        if batch.keys() != self.batch.keys():
            raise ValueError("CUDA graph batch fields differ from the captured batch")
        for name, value in batch.items():
            target = self.batch[name]
            if (value.shape, value.dtype, value.device) != (target.shape, target.dtype, target.device):
                raise ValueError(f"CUDA graph batch shape/dtype/device changed for {name}")
            target.copy_(value)
        self.graph.replay()
        self.agent.step_lr_schedules()
        if not log_stats:
            return {}
        values = self.metrics.detach().cpu().tolist()
        return dict(zip(self.metric_names, values))
