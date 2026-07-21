from __future__ import annotations

import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from run_sam3_train import (  # noqa: E402
    DisplayState,
    HeadlineLoss,
    TrainingProgress,
    apply_yaml_overrides,
    archive_checkpoint_dir,
    build_status_line,
    extract_headline_loss,
    extract_progress,
    materialize_runtime_config,
    parse_config_metadata,
    update_state_from_line,
)


def test_extract_progress_from_sam3_train_line() -> None:
    line = (
        "INFO train_utils.py: Train Epoch: [2][7/31] | Batch Time: 1.0 | "
        "Losses/train_pal_video_train_loss: 7.50e+01 (8.0e+01)"
    )

    progress = extract_progress(line)

    assert progress == TrainingProgress(
        phase="train",
        epoch=2,
        step=7,
        total_steps=31,
    )


def test_exact_sam3_info_line_updates_progress_state() -> None:
    line = (
        "INFO 2026-07-11 21:59:08,505 train_utils.py: 269: "
        "Train Epoch: [0][130/166] | Batch Time: 3.04 (3.54) | "
        "Data Time: 0.00 (0.41) | Mem (GB): 35.00 (34.96/35.00) | "
        "Time Elapsed: 00d 00h 07m | "
        "Losses/train_pal_video_train_loss: 2.22e+02 (1.54e+02) | "
        "Losses/train_default_loss: 0.00e+00 (0.00e+00)"
    )
    state = DisplayState()

    updated = update_state_from_line(state, line)

    assert updated is True
    assert state.progress == TrainingProgress(
        phase="train",
        epoch=0,
        step=130,
        total_steps=166,
    )
    assert state.loss == HeadlineLoss(
        name="train_pal_video_train_loss",
        value=222.0,
    )
    assert state.memory_gb == 35.0


def test_losses_and_meters_line_marks_epoch_complete() -> None:
    line = (
        "INFO trainer.py: Losses and meters: "
        "{'Losses/train_pal_video_train_loss': 140.036, "
        "'Trainer/epoch': 0, 'Trainer/steps_train': 166}"
    )

    progress = extract_progress(line)

    assert progress == TrainingProgress(
        phase="train",
        epoch=0,
        step=165,
        total_steps=166,
    )


def test_extract_headline_loss_from_train_line() -> None:
    line = (
        "INFO train_utils.py: Train Epoch: [0][0/2] | "
        "Losses/train_pal_video_train_loss: 1.68e+02 (1.68e+02) | "
        "Losses/train_pal_video_train_loss_bbox: 0.03 (0.03)"
    )

    loss = extract_headline_loss(line)

    assert loss == HeadlineLoss(name="train_pal_video_train_loss", value=168.0)


def test_extract_headline_loss_from_losses_and_meters_dict() -> None:
    line = (
        "INFO trainer.py: Losses and meters: "
        "{'Losses/train_pal_video_train_loss': 75.175, "
        "'Losses/train_default_loss': 0, "
        "'Losses/train_pal_video_train_presence_loss': 0.001}"
    )

    loss = extract_headline_loss(line)

    assert loss == HeadlineLoss(name="train_pal_video_train_loss", value=75.175)


def test_parse_config_metadata_reads_epochs_and_log_dir(tmp_path: Path) -> None:
    config = tmp_path / "pokeworks_cashier_ft.yaml"
    config.write_text(
        """
paths:
  experiment_log_dir: "/tmp/pal/sam3_finetune/logs/pokeworks_cashier_ft"
scratch:
  max_data_epochs: 20
trainer:
  max_epochs: ${scratch.max_data_epochs}
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
""",
        encoding="utf-8",
    )

    metadata = parse_config_metadata(config)

    assert metadata.max_epochs == 20
    assert metadata.experiment_log_dir == Path(
        "/tmp/pal/sam3_finetune/logs/pokeworks_cashier_ft"
    )
    assert metadata.checkpoint_dir == Path(
        "/tmp/pal/sam3_finetune/logs/pokeworks_cashier_ft/checkpoints"
    )


def test_build_status_line_keeps_terminal_compact() -> None:
    state = DisplayState(
        progress=TrainingProgress(phase="train", epoch=1, step=4, total_steps=10),
        loss=HeadlineLoss(name="train_pal_video_train_loss", value=75.175),
        memory_gb=15.0,
    )

    line = build_status_line(state, max_epochs=20, terminal_width=72)

    assert "train epoch 2/20" in line
    assert "[4/10]" in line
    assert "15/200" in line
    assert "loss=75.2" in line
    assert "mem=15.0G" in line
    assert len(line) <= 72


def test_build_status_line_shows_sam3_epoch_step_token() -> None:
    state = DisplayState(
        progress=TrainingProgress(phase="train", epoch=0, step=130, total_steps=166),
        loss=HeadlineLoss(name="train_pal_video_train_loss", value=222.0),
        memory_gb=35.0,
    )

    line = build_status_line(state, max_epochs=20, terminal_width=100)

    assert "[130/166]" in line
    assert "total=131/3320" in line


def test_archive_checkpoint_dir_moves_existing_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "logs" / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint.write_text("weights", encoding="utf-8")

    archived = archive_checkpoint_dir(checkpoint_dir)

    assert archived is not None
    assert archived.exists()
    assert not checkpoint_dir.exists()
    assert (archived / "checkpoint.pt").read_text(encoding="utf-8") == "weights"


def test_apply_yaml_overrides_replaces_existing_scalar_values() -> None:
    text = """
scratch:
  train_batch_size: 1
  val_batch_size: 1
paths:
  experiment_log_dir: "/tmp/old"
"""

    updated = apply_yaml_overrides(
        text,
        [
            "scratch.train_batch_size=4",
            "scratch.val_batch_size=2",
            "paths.experiment_log_dir=/tmp/new run",
        ],
    )

    assert "  train_batch_size: 4" in updated
    assert "  val_batch_size: 2" in updated
    assert '  experiment_log_dir: "/tmp/new run"' in updated


def test_materialize_runtime_config_writes_patched_sam3_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pal" / "pokeworks_cashier_ft.yaml"
    source.parent.mkdir()
    source.write_text(
        """
scratch:
  train_batch_size: 1
  val_batch_size: 1
""",
        encoding="utf-8",
    )
    cwd = tmp_path / "sam3_repo"
    command = [
        "python",
        "sam3/train/train.py",
        "-c",
        "configs/pal/pokeworks_cashier_ft.yaml",
        "--use-cluster",
        "0",
    ]

    updated_command, patched_path = materialize_runtime_config(
        command=command,
        source_config_path=source,
        cwd=cwd,
        overrides=["scratch.train_batch_size=4"],
    )

    assert patched_path.exists()
    assert patched_path.parent == cwd / "sam3" / "train" / "configs" / "pal"
    assert "train_batch_size: 4" in patched_path.read_text(encoding="utf-8")
    assert updated_command[3].startswith("configs/pal/pokeworks_cashier_ft_runtime_")
    assert updated_command[3].endswith(".yaml")
