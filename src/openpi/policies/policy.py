from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        t_all = time.monotonic()

        t0 = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        input_transform_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        stack_ms = (time.monotonic() - t0) * 1000

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        t0 = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # Force real compute-completion so the timer reflects GPU work, not
        # dispatch latency. JAX execution is async; PyTorch's .cpu() below
        # already blocks so we only need an explicit barrier for JAX.
        if not self._is_pytorch_model:
            jax.block_until_ready(actions)
        model_ms = (time.monotonic() - t0) * 1000

        outputs = {"state": inputs["state"], "actions": actions}

        t0 = time.monotonic()
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        to_numpy_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        outputs = self._output_transform(outputs)
        output_transform_ms = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - t_all) * 1000
        outputs["policy_timing"] = {
            "infer_ms": model_ms,  # kept for back-compat
            "model_ms": model_ms,
            "input_transform_ms": input_transform_ms,
            "stack_ms": stack_ms,
            "to_numpy_ms": to_numpy_ms,
            "output_transform_ms": output_transform_ms,
            "total_ms": total_ms,
            "batch_size": 1,
        }
        return outputs

    def infer_batch(self, obs_list: list[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        """Batched counterpart to :meth:`infer`.

        Applies input transforms per sample (transforms are written for
        unbatched data), stacks along a leading batch axis of N, runs
        ``sample_actions`` once, then splits the outputs back to N
        per-sample dicts with output transforms applied individually.

        Args:
            obs_list: List of N unbatched observation dicts.
            noise: Optional pre-sampled noise. Accepted shapes are
                ``(action_horizon, action_dim)`` (broadcast across the
                batch) or ``(N, action_horizon, action_dim)``.

        Returns:
            List of N dicts shaped like :meth:`infer`'s return value,
            each carrying its own ``policy_timing`` (same model-wall-time
            for all samples; retained per-sample so downstream telemetry
            keeps the single-infer contract).
        """
        if not obs_list:
            return []

        batch_size = len(obs_list)
        t_all = time.monotonic()

        t0 = time.monotonic()
        transformed = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]
        input_transform_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        if not self._is_pytorch_model:
            # Stack on the host as a single numpy op, then ship to device once
            # per leaf. This avoids N individual host->device copies that
            # `jnp.stack([jnp.asarray(x), ...])` would incur.
            inputs = jax.tree.map(
                lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs], axis=0)),
                *transformed,
            )
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda *xs: torch.stack(
                    [torch.from_numpy(np.array(x)).to(self._pytorch_device) for x in xs], dim=0
                ),
                *transformed,
            )
            sample_rng_or_pytorch_device = self._pytorch_device
        stack_ms = (time.monotonic() - t0) * 1000

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_arr = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )
            if noise_arr.ndim == 2:
                if self._is_pytorch_model:
                    noise_arr = noise_arr.unsqueeze(0).expand(batch_size, *noise_arr.shape)
                else:
                    noise_arr = jnp.broadcast_to(noise_arr[None, ...], (batch_size, *noise_arr.shape))
            if noise_arr.shape[0] != batch_size:
                raise ValueError(
                    f"noise batch dim ({noise_arr.shape[0]}) does not match obs batch dim ({batch_size})"
                )
            sample_kwargs["noise"] = noise_arr

        observation = _model.Observation.from_dict(inputs)

        t0 = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if not self._is_pytorch_model:
            jax.block_until_ready(actions)
        model_ms = (time.monotonic() - t0) * 1000

        batched_outputs = {"state": inputs["state"], "actions": actions}

        t0 = time.monotonic()
        if self._is_pytorch_model:
            batched_np = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), batched_outputs)
        else:
            batched_np = jax.tree.map(lambda x: np.asarray(x), batched_outputs)
        to_numpy_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        results: list[dict] = []
        for i in range(batch_size):
            sample_out = jax.tree.map(lambda x: x[i, ...], batched_np)
            sample_out = self._output_transform(sample_out)
            results.append(sample_out)
        output_transform_ms = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - t_all) * 1000
        timing = {
            "infer_ms": model_ms,  # kept for back-compat
            "model_ms": model_ms,
            "input_transform_ms": input_transform_ms,
            "stack_ms": stack_ms,
            "to_numpy_ms": to_numpy_ms,
            "output_transform_ms": output_transform_ms,
            "total_ms": total_ms,
            "batch_size": batch_size,
        }
        for sample_out in results:
            sample_out["policy_timing"] = dict(timing)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def infer_batch(self, obs_list: list[dict]) -> list[dict]:
        # Forward to the wrapped policy's batched path if available; otherwise
        # fall back to a sequential loop of recorded single-infer calls.
        if hasattr(self._policy, "infer_batch"):
            results = self._policy.infer_batch(obs_list)
            for obs, res in zip(obs_list, results, strict=True):
                data = {"inputs": obs, "outputs": res}
                data = flax.traverse_util.flatten_dict(data, sep="/")
                output_path = self._record_dir / f"step_{self._record_step}"
                self._record_step += 1
                np.save(output_path, np.asarray(data))
            return results
        return [self.infer(obs) for obs in obs_list]
