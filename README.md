# RaCE-PV — research source library

Current private development source, synchronized on 2026-10-05.
This is a source-only repository, not a complete data/weights/experiment artifact.
The previous legacy extract remains available in Git history.

## 中文说明

本次更新包含已核对的 160 个模型库源码文件（runtime v4），包括可靠性门控、
独立目标函数、C/F 研究变体及训练组件；另包含 2026-10-02 C 短试训的
配置策略和薄适配层。源码更新不改变已交付的实验结果或模型排名。

不上传原始数据、预测数组、检查点、SSH 配置、云端控制脚本、运行记录、
聊天附件或本地环境。库文件逐字节保留，不改动正在运行的训练。

## Included

- 160 Python library files under `src/racepv`, matched byte-for-byte against
  the runtime-v4 source archive and the recovered source reference.
- Seven synthetic model/loss/identity test modules.
- Four historical YAML configurations, packaging metadata and license notices.
- `experiments/c_pilot`: fixed C0/C-Reg/C-Temp policy, model/trainer adapter,
  and four synthetic policy checks. No execution controller or private binding.

`SNAPSHOT.json` records copied-source provenance; `SHA256SUMS` covers the
published files except itself. See [VERIFICATION.md](VERIFICATION.md) for the
actual verification scope and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Local CPU verification

Use an isolated Python 3.12 environment; do not modify a running experiment's
environment. Install an appropriate CPU PyTorch build, then:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py
```

Tests use synthetic tensors and explicit CPU/GRU paths. Four source-equivalence
checks require the separately supplied original-experts reference and are
skipped when `RACEPV_ORIGINAL_EXPERTS` is unset. CPU checks do not establish
native Mamba/CUDA equivalence, benchmark accuracy, or full reproducibility.

The configuration files are historical examples, not the resolved contract of
every C/F experiment. Some library CLI routes require omitted scripts, qualified
data and external dependencies. Do not treat them as ready-to-run full campaigns.

## C development pilot

See [the pilot source notes](experiments/c_pilot/README.md). C0 uses weight decay
1e-4 and router temperature 1; C-Reg changes weight decay to 1e-3; C-Temp changes
the shared router temperature to 1.25. Other approved conditions stay fixed.
These are development variants, not a replacement final-test leaderboard.

## Four C-family architecture explorations

The actual historical C2, C3-TimeXer-PV, C-LG and C-DH-linear development
implementations are now available under `src/racepv/exploratory`, including
their narrow training adapters and synthetic tests. They are separate from
the original final-model registry; none replaces C, F or M1 automatically.
See [variant usage and development status](experiments/c_variants/README.md)
and [source provenance](experiments/c_variants/SOURCE_PROVENANCE.json).
No real-data training or scoring was run for this GitHub update.

## License and source boundaries

The existing [MIT license](LICENSE) and retained upstream notices remain.
[THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) distinguishes local adaptations
from external implementations, weights and data. Repository visibility is not
changed by this synchronization. The earlier rights statement concerned the
historical extract; it is not an independent legal certification of this update.
