"""Tests for src.pinn.run_summary."""


def test_import():
    from src.pinn import run_summary  # noqa: F401

    assert hasattr(run_summary, "format_run_summary")
    assert hasattr(run_summary, "count_parameters")


def test_count_parameters_matches_default_pinn():
    from src.pinn.model import PINN
    from src.pinn.run_summary import count_parameters

    model = PINN(
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
        input_dim=3,
        output_dim=2,
        hidden_layers=5,
        hidden_width=100,
    )
    # (3+1)*100 + 4*(100+1)*100 + (100+1)*2 -- buffers (lower_bound,
    # upper_bound) must NOT be counted, or this would read 41008.
    assert count_parameters(model.state_dict()) == 41002


_ARGS = {
    "hidden_layers": 5,
    "hidden_width": 100,
    "x_max": 1.0,
    "y_max": 1.0,
    "t_max": 0.005,
    "epsilon": 0.01,
    "pde_weight": 1.0,
    "ic_weight": 1.0,
    "bc_weight": 1.0,
    "energy_penalty": False,
    "energy_weight": 1.0,
    "epochs": 100,
    "lr": 0.001,
    "lbfgs_steps": 10,
    "device": "cpu",
    "run_name": "demo",
    "smoke_test": False,
    "checkpoint_every": 500,
    "log_every": 100,
}


def test_format_run_summary_bare_args():
    from src.pinn.run_summary import format_run_summary

    text = format_run_summary(_ARGS)
    assert "hidden_width: 100" in text
    assert "device" not in text
    assert "run summary" not in text  # header block only appears with kwargs


def test_format_run_summary_with_model_and_history():
    from src.pinn.model import PINN
    from src.pinn.run_summary import format_run_summary

    model = PINN(
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
        input_dim=3,
        output_dim=2,
        hidden_layers=5,
        hidden_width=100,
    )
    smoke_args = dict(_ARGS, smoke_test=True)
    history = {
        "pde": [1.0, 0.5, 0.1, 0.05, 0.01],
        "ic": [1.0, 0.9, 0.8, 0.7, 0.6],
        "bc": [0.5, 0.4, 0.3, 0.2, 0.1],
        "total": [2.5, 1.8, 1.2, 0.95, 0.71],
    }

    text = format_run_summary(
        smoke_args, model_state=model.state_dict(), history=history, source="checkpoint.pt"
    )
    assert "parameters: 41002" in text
    assert "source: checkpoint.pt" in text
    assert "adam steps: 5" in text
    assert "final total: 7.100000e-01" in text
