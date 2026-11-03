"""CPU contract checks for the isolated history residual adapter."""
import json

import torch

from research.history_stream.residual import HistoryResidualAdapter


def main():
    torch.manual_seed(20261103)
    model = HistoryResidualAdapter(feature_dim=16, hidden_dim=8)
    native = torch.randn(2, 5, 16)
    history = torch.randn(2, 5, 16)
    age = torch.tensor([[1.], [64.]])
    available = torch.tensor([[0.], [1.]])

    initial = model(native, history, age, available)
    assert torch.equal(initial, native)

    with torch.no_grad():
        model.residual[-1].weight.normal_(std=.05)
        model.residual[-1].bias.normal_(std=.05)
    output = model(native, history, age, available)
    assert torch.equal(output[1], native[1])
    assert not torch.equal(output[0], native[0])

    changed_history = history.clone()
    changed_history[0] += 1.
    changed_age = age.clone()
    changed_age[0] += 1.
    assert not torch.equal(model(native, changed_history, age, available)[0], output[0])
    assert not torch.equal(model(native, history, changed_age, available)[0], output[0])
    assert torch.equal(model(native, changed_history, age, available)[1], native[1])
    assert torch.equal(model(native, history, changed_age, available)[1], native[1])

    model.zero_grad()
    target = torch.randn_like(native)
    loss = (model(native, history, age, available)[0] - target[0]).square().mean()
    loss.backward()
    gradient = model.residual[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.

    print(json.dumps(dict(
        status='PASS',
        zero_initialized_native_parity=True,
        available_stream_unchanged_after_learning=True,
        missing_stream_uses_history_and_age=True,
        output_projection_receives_finite_gradient=True,
        cuda_initialized=torch.cuda.is_initialized(),
        scope='CPU tensor contract only; no model integration or efficacy evidence'),
        sort_keys=True))


if __name__ == '__main__':
    main()
