"""Tests for the retinal-response command-line interface."""

from types import SimpleNamespace

from retinamodel import retina_model_cli


def test_build_parser_defaults():
    """Check the build command parses its required and default arguments."""

    parser = retina_model_cli.build_parser()

    args = parser.parse_args(
        [
            "build",
            "--dataset",
            "data/interm/retina_dataset_test",
        ]
    )

    assert args.command == "build"
    assert args.dataset == "data/interm/retina_dataset_test"
    assert args.config == "config/retina_model.toml"
    assert args.output_dir == "data/interm"
    assert args.func is retina_model_cli._build_command


def test_build_command_calls_retina_model(monkeypatch, capsys):
    """Check the CLI delegates the build to the Module 4 pipeline."""

    dataset = SimpleNamespace(
        dataset_dir="data/interm/retina_dataset_test",
        outs=SimpleNamespace(
            shape=(2, 9, 4, 4)
        ),
    )

    sizes = {
        "extrema_mask_gib": 0.001,
        "outs_fill_gib": 0.004,
        "total_gib": 0.005,
    }

    paths = SimpleNamespace(
        run_dir="data/interm/retina_model_test",
        outputs_zarr=(
            "data/interm/retina_model_test/"
            "retina_outputs.zarr"
        ),
        metadata_csv=(
            "data/interm/retina_model_test/"
            "metadata.csv"
        ),
        processing_settings_json=(
            "data/interm/retina_model_test/"
            "processing_settings.json"
        ),
    )

    calls = {}

    def fake_load(dataset_path):
        calls["load"] = dataset_path
        return dataset

    def fake_estimate(input_dataset):
        calls["estimate"] = input_dataset
        return sizes

    def fake_build(
        dataset_dir,
        config_path,
        base_dir,
        show_progress,
    ):
        calls["build"] = {
            "dataset_dir": dataset_dir,
            "config_path": config_path,
            "base_dir": base_dir,
            "show_progress": show_progress,
        }
        return paths

    monkeypatch.setattr(
        retina_model_cli,
        "load_retina_input_dataset",
        fake_load,
    )
    monkeypatch.setattr(
        retina_model_cli,
        "estimate_retina_model_storage",
        fake_estimate,
    )
    monkeypatch.setattr(
        retina_model_cli,
        "build_retina_model",
        fake_build,
    )

    args = SimpleNamespace(
        dataset="data/interm/retina_dataset_test",
        config="config/retina_model.toml",
        output_dir="data/interm",
    )

    retina_model_cli._build_command(args)

    assert calls["load"] == args.dataset
    assert calls["estimate"] is dataset

    assert calls["build"] == {
        "dataset_dir": args.dataset,
        "config_path": args.config,
        "base_dir": args.output_dir,
        "show_progress": True,
    }

    output = capsys.readouterr().out

    assert "Retinal-response build" in output
    assert "Retinal-response build complete" in output
    assert "retina_model_test" in output


def test_view_parser_defaults():
    """Check the view command parses its required and default arguments."""

    parser = retina_model_cli.build_parser()

    args = parser.parse_args(
        [
            "view",
            "--run",
            "data/interm/retina_model_test",
        ]
    )

    assert args.command == "view"
    assert args.run == "data/interm/retina_model_test"
    assert args.config == "config/retina_model.toml"
    assert args.func is retina_model_cli._view_command


def test_view_command_launches_viewer(monkeypatch):
    """Check the CLI delegates viewing without importing Tk during build."""

    calls = {}

    def fake_launch(run_dir, config_path):
        calls["run_dir"] = run_dir
        calls["config_path"] = config_path

    monkeypatch.setattr(
        retina_model_cli,
        "_launch_viewer",
        fake_launch,
    )

    args = SimpleNamespace(
        run="data/interm/retina_model_test",
        config="config/retina_model.toml",
    )

    retina_model_cli._view_command(
        args
    )

    assert calls == {
        "run_dir": "data/interm/retina_model_test",
        "config_path": "config/retina_model.toml",
    }