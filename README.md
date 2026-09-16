# RaCE-PV — legacy source-library snapshot

**Local release draft, not yet published. Historical research code, not the final paper artifact.**

This is a byte-preserving subset of the source snapshot taken on 2026-09-15,
before the reliability-gate A candidate was added. It includes the original
three-expert probabilistic photovoltaic forecasting model, its losses, supporting
Python library, four historical configuration files, and three synthetic test
modules. The A candidate and its independent objective are not included.

## 中文说明

这是旧版代码公开前的整理草稿，不是当前改进版，也不是完整实验复现包。
保留原模型和原配置，不将旧结果重新包装为新结果。`configs/base.yaml`
中的 `specialist_loss_weight: 0.15` 是历史默认值，不代表当前最终模型决定，
也不能据此认定任意历史实验都使用了这一配置。

已知数据时间口径、观测/插值来源和评估限制应单独核实；本包不声称这些问题
已经解决，不提供性能排名、SOTA 声明或最终论文结论。见
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

## Included / excluded

- Included: 130 library Python files, 3 synthetic test modules, 4 original YAML
  configurations, original `pyproject.toml`, existing project MIT `LICENSE`.
- Excluded: datasets, calibration/test arrays, predictions, checkpoints, run
  directories, cloud-operation scripts, manuscript/PPT, chat attachments,
  environment directories, third-party checkouts, credentials and Git history.
- Original copied file identities are listed in `SNAPSHOT.json`. `SHA256SUMS`
  records this assembled release draft (excluding the checksum file itself).
- The wider historical scripts and fixtures are intentionally omitted. Only the
  CPU synthetic model/loss checks below are the supported validation route for
  this extract. Do not assume all historical CLI workflows are self-contained.

## Local CPU check (no dataset, no training)

Use a separate Python 3.12 environment. Do not install this historical package
over the environment running a current experiment. The original dependency
ranges are retained; this extract does not provide a reproducible environment
lock or claim that every dependency version satisfying those ranges was tested.

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_model.py tests/test_ramp_refinement_losses.py tests/test_sparse_tail_losses.py
```

The selected tests use synthetic tensors and explicit GRU/other CPU test paths;
they do not establish official Mamba/CUDA equivalence or prediction performance.
No GPU, raw data, calibration, or heldout evaluation is needed for these checks.
The tests perform forward/backward checks, not an experimental training run.

This draft passed 71 synthetic tests. Static lint passes with the original
first-party classification recorded in `ruff.toml`; format check still reports
57 historical files. Source bytes were not reformatted. See
[VERIFICATION.md](VERIFICATION.md) for the exact scope and limitations.

## Reuse and release status

`LICENSE` preserves the existing project's MIT notice; it was copied separately
and is identified in `SNAPSHOT.json`. Public release still requires confirmation
that the owner/contributors can distribute this code and that any adapted
third-party portions have all required notices. An MIT notice for this project
does not grant redistribution rights to datasets or external dependencies.
See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md).

No GitHub repository or remote was created by this extraction. If publishing
later, start from **this directory only**, not its parent workspace or audit
directory, and do not import the original project's Git history.
