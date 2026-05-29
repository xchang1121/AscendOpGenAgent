# AutoResearch

Claude Code 驱动的 Triton-Ascend 算子自动优化框架。固定栈：Triton-Ascend kernel + Ascend NPU + PyTorch ref。

首次使用按 [路径 A](#路径-a本机--测评机) 或 [路径 B](#路径-b本机-dev--远程-npu-测评机) 顺序执行。其它情况按下表索引跳转。

## 章节索引

| 主题 | 章节 |
|------|------|
| 命令参数 | [/autoresearch 完整参数](#autoresearch-完整参数) |
| 文件命名约束 | [命名契约](#命名契约) |
| 单轮 phase 流程 | [主循环](#主循环) |
| eval 执行与断连处理 | [Eval 执行](#eval-执行) |
| 精度判定 | [精度容差](#精度容差) |
| 状态文件位置 | [配置与状态](#配置与状态) |
| 环境与 env.sh | [环境](#环境) |
| 远端 worker ar_cli 细节 | [远程 worker](#远程-worker) |
| 批量运行 | [批量跑](#批量跑) |
| Skills 调优手册 | [Skills 库](#skills-库) |
| Hook 与内部接线 | [内部机制](#内部机制) |

---

## 路径 A：本机 = 测评机

开发与 eval 位于同一台 Ascend NPU 机器，配置最少。

### A.1 准备环境

前置：
- 本机具备 Ascend NPU，`npu-smi info` 可列出设备
- 本机已具备可 import torch_npu 与 triton 的 Python 环境（conda、venv、系统 Python 均可）
- 已安装 Claude Code CLI

操作：编写 `~/env.sh`，使非交互 shell `source` 之后可直接执行 `python -c "import torch_npu, triton"`。常用做法为 source CANN 的 `set_env.sh` 并激活已有的 Python 环境。

详见 [环境](#环境)。

验证：
```bash
source ~/env.sh
python -c "import torch_npu, triton"
npu-smi info
```

### A.2 准备 ref 与 kernel

前置：A.1 完成。

操作：在 `autoresearch/workspace/` 下创建以下两个文件。

`workspace/relu_ref.py`（PyTorch 标准答案）：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x): return F.relu(x)

def get_inputs():      return [torch.randn(1024, 1024, dtype=torch.float16)]
def get_init_inputs(): return []
```

`workspace/relu_kernel.py`（种子 kernel，必须包含 `@triton.jit` 并实际 launch，否则 quick_check 拒绝）：

```python
import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        if not x.is_npu: x = x.npu()
        x = x.contiguous()
        out = torch.empty_like(x)
        n, BLOCK = x.numel(), 1024
        _relu_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
        return out
```

命名硬约束见 [命名契约](#命名契约)；违反将导致流程无法启动。

### A.3 启动 /autoresearch

前置：A.1、A.2 完成。

```bash
cd autoresearch
source ~/env.sh
claude
```

在 Claude prompt 中输入：

```text
/autoresearch --ref workspace/relu_ref.py --kernel workspace/relu_kernel.py \
  --op-name relu --devices 0 --max-rounds 5
```

`--devices` 必填，取值为本机 NPU 卡下标。其余 flag 见 [/autoresearch 完整参数](#autoresearch-完整参数)。

### A.4 进度查看

Claude 按 scaffold → BASELINE → PLAN → EDIT → eval → KEEP/DISCARD → 下一轮自动推进，无需人工干预，按 `[AR Phase: ...]` 提示继续。

另开终端实时查看：

```bash
cd autoresearch
python scripts/dashboard.py ar_tasks/relu_<ts>_<hex>/ --watch
```

阶段流转见 [主循环](#主循环)。

### A.5 结果

达到 `--max-rounds` 后自动进入 FINISH，写出以下文件：
- `<task_dir>/kernel.py`：最佳 kernel
- `<task_dir>/.ar_state/report.md`：报告（内嵌 SVG）

每个 KEEP 对应一次 git commit。

---

## 路径 B：本机 dev + 远程 NPU 测评机

适用：本机无 NPU，eval 在远端 Ascend 机器执行，orchestrator 留在本机。

### B.1 远端准备环境

操作与 [A.1](#a1-准备环境) 相同，执行位置为远端 NPU 机器。

### B.2 本机准备

前置：本机可 SSH 至远端，`~/.ssh/config` 已配置 alias（后文以 `my-npu` 为例）。

操作：本机仅需安装 Python ≥ 3.10、PyYAML、Claude Code CLI，无需 torch_npu / CANN。

### B.3 配置 worker host

前置：B.1、B.2 完成。

编辑 `autoresearch/config.yaml`：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path: /home/<user>/AscendOpGenAgent/autoresearch
      env_script: /home/<user>/env.sh
```

字段语义与可选项见 [远程 worker](#远程-worker)。

### B.4 启动远端 worker 与 tunnel

```bash
cd autoresearch
python scripts/ar_cli.py worker --remote-host my-npu --start \
    --backend ascend --arch ascend910b3 --devices 6 --port 9111
```

该命令完成两项操作：通过 SSH 在 my-npu 启动 daemon；本机启动 `ssh -L 9111:127.0.0.1:9111`。后续对 worker 的访问均经此 tunnel 透传。

验证：

```bash
python scripts/ar_cli.py worker --remote-host my-npu --status --port 9111
```

### B.5 准备 ref 与 kernel

同 [A.2](#a2-准备-ref-与-kernel)，文件位于 `autoresearch/workspace/`。

### B.6 启动 /autoresearch（携带 --worker-url）

```bash
cd autoresearch
claude
```

```text
/autoresearch --ref workspace/relu_ref.py --kernel workspace/relu_kernel.py \
  --op-name relu --devices 6 --max-rounds 5 --worker-url 127.0.0.1:9111
```

`--devices` 取值为**远端** NPU 卡下标。`--worker-url` 由 scaffold 透传写入 `task.yaml: worker.urls`，后续每轮 eval 自动经远端执行。

### B.7 进度与结果

同 [A.4](#a4-进度查看)、[A.5](#a5-结果)。

### B.8 停止

不再使用时执行：

```bash
python scripts/ar_cli.py worker --remote-host my-npu --stop --port 9111
```

---

## 续跑、重置与历史查询

路径 A / B 均适用：

| 操作 | 命令 |
|------|------|
| 续跑最近 task | `/autoresearch --resume` |
| 续跑指定 task | `/autoresearch --resume <task_dir>` |
| 重新开始 | 删除 `ar_tasks/<task_dir>/`，再执行 `/autoresearch --ref ... --kernel ...` |
| 查看每轮记录 | `<task_dir>/.ar_state/history.jsonl`，一行一轮 |
| 查看 plan 当前状态 | `<task_dir>/.ar_state/plan.md` |

---

## /autoresearch 完整参数

入参语义：
- `--`-prefixed flag：新建任务（scaffold + 首次 baseline 原子完成）
- 已存在的目录路径：resume 该目录
- `--resume`：resume 最近活跃 task
- 无参数：进入交互式询问

新建仅接受一种入口：`--ref X.py --kernel Y.py`。scaffold 自动运行 baseline：seed PASS 后进入 PLAN；seed FAIL 同样进入 PLAN，首批 plan items 用于改写 seed kernel。

CLI flag：

| flag | 取值 | 说明 |
|------|------|------|
| `--devices` | 整数列表（`5` 或 `0,1,2,3`） | 必填 |
| `--max-rounds` | 整数 | 默认 30 |
| `--eval-timeout` | 整数（秒/shape） | 默认 600 |
| `--no-code-checker` | flag | 关闭 [`engine/quick_check.py`](scripts/engine/quick_check.py) 的 Triton 退化 AST 检查 |
| `--worker-url` | `host:port[,host:port]` | 经远端 worker 执行 eval |

`arch`（如 `ascend910b3`）由 `--devices` 选中的卡 `npu-smi info` 推断，写入 task.yaml 仅用于 dashboard 与 report 显示。

## 命名契约

下列约束为硬性要求，违反将导致流程无法启动。

| 项 | 约束 |
|----|-----|
| ref 文件名 | 单跑任意；批跑严格 `<op>_ref.py`（batch/prepare 按 stem 配对） |
| kernel 文件名 | 单跑任意；批跑严格 `<op>_kernel.py` |
| ref 暴露 | `class Model(nn.Module)` + `get_init_inputs()`，并二选一：`get_inputs()` 单 shape / `get_input_groups()` 多 shape |
| kernel 暴露 | `class ModelNew(nn.Module)`；必须包含 `@triton.jit` kernel 并实际 launch，且 forward 路径不得通过 `torch.*` / `F.*` / tensor 方法 / `@` 算子 / Python 循环完成计算（否则 [`engine/quick_check.py`](scripts/engine/quick_check.py) 拒绝，规则定义于 [`skills/triton/kernel-verifier/scripts/validate_triton_impl.py`](../skills/triton/kernel-verifier/scripts/validate_triton_impl.py)） |
| sibling shape/cache | 文件 stem 必须以 ref stem 为前缀（ref `10_Relu.py` 对应 `10_Relu*.json/.pt/.npz`）；scaffold 自动拷入 task_dir 并写入 `task.yaml: data_files` |
| task.yaml 字段 | fixed schema（[loader.py](scripts/task_config/loader.py)）；拼写错误（如 `editable_file`）被 yaml 默认值覆盖，手动修改需谨慎 |
| task_dir 命名 | `ar_tasks/<op_name>_<int(time.time())>_<uuid.hex[:6]>/`，batch/manifest 按此三段识别，禁止手动修改 |

## 主循环

单轮：PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle。
连续 3 次 FAIL 转入 DIAGNOSE，plan 全部 settle 转入 REPLAN，预算耗尽转入 FINISH。

```
INIT
  │  /autoresearch --ref X.py --kernel Y.py
  ▼
BASELINE  (scaffold --run-baseline 原子完成，运行 seed kernel)
  │
  ▼
PLAN  (BASELINE PASS 或 FAIL 均进入 PLAN；FAIL 时首批 plan items 改写 seed)
  │  create_plan.py 校验
  ▼
   ┌────────────────────────────────── EDIT ◀────────────┐
   │  pipeline.py:                                       │
   │    quick_check → run_eval → keep_or_discard         │
   │   ├─ KEEP    : git commit (editable_files)，更新 best│
   │   ├─ DISCARD : 回滚 editable_files                   │
   │   └─ FAIL    : consecutive_failures++，回滚         │
   │                                                     │
   │   ├─ failures ≥ 3              ─→ DIAGNOSE ─→ PLAN ─┤
   │   ├─ plan 全部 settle           ─→ REPLAN  ─→ PLAN ─┤
   │   └─ eval_rounds == max_rounds ─→ FINISH
   └─────────────────────────────────────────────────────┘
```

DIAGNOSE 与 REPLAN 不绕回 PLAN：`create_plan.py` 校验通过后由 hook 直接写入 `phase = EDIT`。每个 plan item 在 `history.jsonl` 中持有 KEEP / DISCARD / FAIL 终态，或在 REPLAN/DIAGNOSE 边界被静默丢弃；pid 单调递增，不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|------|-------------|------|
| BASELINE | `baseline.py` | seed_metric 写入 progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）|
| EDIT | Edit kernel.py 后运行 `pipeline.py` | history.jsonl，可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG）|

## Eval 执行

`baseline.py` 与 `pipeline.py` 在进程内直接调用 `task_config.run_eval`，按 `worker_urls` 是否非空分两条路径。

本地路径（默认）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(task_dir, config, device_id)
     └─ utils.eval_runner.local_eval(task_dir, ...)            # 2-pass driver
          ├─ Popen eval_kernel.py --phases profile_base ...    # ref 单独运行
          └─ Popen eval_kernel.py --phases verify,profile_gen  # kernel 运行
```

远程路径（`--worker-url` 或 `task.yaml worker.urls`）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(..., worker_urls=[...])
     ├─ package_builder.build_package         # tar.gz: task.yaml + ref + kernel + data_files
     └─ HTTP POST /api/v1/run → worker        # multipart
                                  └─ worker/server.py 收包 → safe_extract →
                                     utils.eval_runner.local_eval（同上）
```

两条路径在 worker 内部共用同一份 `local_eval`，避免 drift。subprocess 拆为 ref pass 与 kernel pass 是为避免 kernel 端 SIGKILL 或设备 hang 影响已落盘的 ref 时延。

`eval_kernel.py` 为静态脚本（不再 codegen），在同一进程内按请求顺序串行执行 phase。verify 之后执行 profile_gen，triton JIT 编译缓存在 profile_gen 阶段保持热态。直接调试：

```bash
cd <task_dir>
python <repo>/autoresearch/scripts/engine/eval_kernel.py \
    --task-dir . --op-name <op> --kernel-file kernel --ref-file reference \
    --device-id 0 \
    --phases verify,profile_gen
```

verify 失败时 ref 时延由 profile_base 在同进程下一段单独测得，与 verify 解耦，dashboard 顶栏始终显示 PyTorch baseline。Sticky baseline 写定（`baseline_metric` 与 `baseline_source=ref` 写入 progress.json）后，后续轮的 phase 列表不再包含 `profile_base`。

### 客户端断连自动 cancel

worker `/api/v1/run` 同时 await eval task 与 disconnect watch task：
- eval 先完成：watch 取消，返回结果
- 客户端先断开（例如 `claude --print` 因 wall-clock 终止）：eval task 取消，subprocess group SIGTERM，释放 device，返回 HTTP 499

由此避免客户端进程终止后 eval 继续占用 device。

非有限浮点（inf、-inf、nan）在 worker 序列化前递归改写为 `null`，避免 FastAPI 默认 JSON 编码器拒收并返回 HTTP 500。

## 精度容差

verify.py（Tier 2 预检）与 `/autoresearch` 每轮 verify 共用 [`correctness.py`](scripts/utils/correctness.py)，对齐 `main` 分支 `skills/triton/kernel-verifier/scripts/verify.py` 的 allclose-style 标准：

- 按 ref dtype 取 `(rtol, atol)`：fp32 → `(2^-13, 1e-5)`，fp16 → `(2^-10, 1e-3)`，bf16 → `(2^-7, 1e-2)`，未知 dtype 回落至 fp32
- 元素级判定：`|new - ref| <= atol + rtol·|ref|`，需逐元素满足
- 额外硬性检查：NaN 位置一致；Inf 位置与符号一致；bool 与 int 精确匹配

## 配置与状态

| 路径 | 用途 |
|------|------|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref 与 kernel 输入 |
| `task.yaml` | 任务配置：name / arch / editable_files / ref_file / devices / eval_timeout / max_rounds / metric / `data_files`（sibling JSON/PT/NPZ，scaffold 自动写入）/ `worker.urls`（远程 worker URL 列表，可选）|
| `.ar_state/.phase` | 当前阶段 |
| `.ar_state/plan.md` | 规划与结算历史（权威态）|
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/progress.json` | 运行时状态 |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（CLAUDE.md 不变量 #10）|
| `config.yaml` | `hallucinated_scripts`（脚本名容错映射）与 `remote_worker.hosts`（SSH alias → repo_path / env_script，供 `ar_cli worker --remote-host` 使用）|
| `.claude/settings.json` | Hook 与权限配置（已提交至仓库）|
| `.claude/settings.local.json` | API key、model 覆盖（不提交至 git）|

`.ar_state/` 内除 `plan_items.xml` 与 `diagnose_v<N>.md`（DIAGNOSE）外均由 hook 与脚本管理，Claude 不可手写。

## 环境

### 测评机依赖

任意 Python 环境（conda、venv、系统 Python 均可），可 import 以下包：

- Python ≥ 3.10、PyYAML
- torch、torch_npu（匹配 CANN 版本）、triton-ascend
- CANN runtime（通过 source `set_env.sh` 等方式加载）

dev 端仅需 Python ≥ 3.10、PyYAML、Claude Code CLI；torch_npu 与 CANN 不需要（eval 全部委托测评机执行）。

### env.sh 契约

若干场景需在非交互 shell 中加载测评 runtime：tmux 非 login shell、`ar_cli worker --remote-host` 远端启动、batch happy-path 的 `bash --login -c 'source env.sh && python ...'`。以上场景共用同一脚本，其路径填入 `config.yaml: remote_worker.hosts.<alias>.env_script`。

env.sh 唯一职责：source 完成后，`python -c "import torch_npu, triton"` 不抛异常。常见写法（按本机 Python 与 CANN 安装方式调整）：

```bash
# ~/env.sh —— 例 1：conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env 名>
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ~/env.sh —— 例 2：venv
source ~/.venvs/<env 名>/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ~/env.sh —— 例 3：系统 Python，仅加载 CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

验证：`source ~/env.sh && python -c "import torch_npu, triton"` 不抛异常。

> env.sh 中禁止使用 `cd`（外部 ar_cli 已执行 `cd repo_path`）、`exec`、`set -e`。

## 远程 worker

[路径 B](#路径-b本机-dev--远程-npu-测评机) 已涵盖用户侧操作流程。以下补充 ar_cli 内部细节：

- `--start --remote-host` 在 dev 端组装 `ssh <alias> 'bash -lc "source env_script && cd repo_path && python scripts/ar_cli.py worker --start --bg ..."'`，远端 daemon log 写入 `/tmp/ar_worker_<port>.log`
- 随后启动 `ssh -f -N -T -L <port>:127.0.0.1:<port> <alias>`，pid 写入 `~/.autoresearch_state/tunnels/<port>.pid`（Windows 使用 `Get-CimInstance` 获取 pid，POSIX 使用 `pgrep`）
- `--stop` 反向操作：先经 ssh 在远端终止 daemon，再 SIGTERM 本机 tunnel
- `--status` 直接 curl `127.0.0.1:<port>/api/v1/status`（经 tunnel 透传）
- 远端 worker `worker/server.py` 仅暴露两个接口：`/api/v1/run`（multipart：tar 包 + task_id + op_name + timeout）与 `/api/v1/status`；`/run` 收包 → safe_extract → 调用与本地路径相同的 2-pass `local_eval` → 返回合并后的 verify/profile dict

打包内容（`task_config/package_builder.py`）：`task.yaml` + `ref_file` + `editable_files[*]` + `data_files[*]`。`data_files` 由 scaffold 在创建 task_dir 时按 ref stem 自动扫描 `.json`、`.pt`、`.npz` 兄弟文件并写入 task.yaml，用于解决多 shape JSON、缓存 .pt 等 sibling 依赖在远端缺失的问题。

config.yaml host 字段：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /home/<user>/AscendOpGenAgent/autoresearch  # 必填
      env_script: /home/<user>/env.sh                         # 必填
      python:     python                                      # 可选，默认 python
      ssh_alias:  my-npu                                      # 可选，默认 = key
```

## 批量跑

`autoresearch/scripts/batch/` 提供对一批 `(ref.py, kernel.py?)` 任务批量执行 `/autoresearch` 的脚本。批级状态（done/error/pending、指向 task_dir 的链接）写入 `<batch_dir>`，round 级状态由 `/autoresearch` 写入 `<repo>/ar_tasks/<op>_<ts>_<uuid>/`。

### happy path

约束：cwd 位于 `autoresearch/` 子目录；远端长时间运行使用 tmux 或 screen；`--devices` 必填，整批串行复用同一张卡。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 0. 切换至子目录
cd autoresearch

# 1. 放置 ref / kernel 文件
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. discover + Tier 1 verify（语法 / import / 必备 export）
python scripts/batch/prepare.py $BATCH_DIR

# 3.（可选）Tier 2 预检：本机实际运行 ref vs kernel
python scripts/batch/verify.py $BATCH_DIR --full

# 4. 后台运行
# tmux 默认 shell 非 login，须使用 bash --login + source env.sh
tmux new -d -s ar_batch \
    "bash --login -c 'source ~/env.sh && \
     python -u scripts/batch/run.py $BATCH_DIR \
       --devices $DEVICE 2>&1 | tee -a $BATCH_DIR/batch.log'"

# 5. 监控（另开终端）
python scripts/batch/monitor.py $BATCH_DIR

# 6. 运行结束后汇总
python scripts/batch/summarize.py $BATCH_DIR
```

### 启动模式

每个 op 需要一对 `(ref.py, kernel.py)`。scaffold `--run-baseline` 运行 seed：
- BASELINE PASS → phase 直接进入 `PLAN` → max-rounds 轮性能优化
- BASELINE FAIL → phase 同样进入 `PLAN`，首批 plan items 改写 seed kernel；连续 3 次失败触发 DIAGNOSE

### Batch 目录布局

```
<batch_dir>/                         ← 批级
  manifest.yaml                      # prepare.py 写入
  batch_progress.json                # run.py 写入：每个 op 的 status / task_dir / metrics
  batch.log                          # run.py 写入：claude --print 的全部 stdout
  verify_results.json                # prepare.py / verify.py 写入
  refs/<op>_ref.py                   # 文件名必须严格为 <op>_ref.py
  kernels/<op>_kernel.py             # 文件名必须严格为 <op>_kernel.py

<repo>/ar_tasks/<op>_<ts>_<uuid>/    ← round 级（由 /autoresearch 维护）
  kernel.py / reference.py / task.yaml
  .ar_state/{.phase, progress.json, plan.md, history.jsonl, report.md}
```

`batch_progress.json:cases.<op>.task_dir` 连接批级与 round 级。

### manifest.yaml

`prepare.py` 自动生成：

```yaml
ref_dir: refs
kernel_dir: kernels
ops:
- op_a
- op_b
```

增删 ref 或 kernel 文件后重新执行 `prepare.py $BATCH_DIR`，ops 列表整体重扫。

### Verify 两档

| Tier | 触发 | 检查 | 需要硬件 |
|------|------|------|----------|
| 1 | `prepare.py` 默认 | 语法 / import / 必备 export（`Model`、`ModelNew`、`get_inputs` 等） | 否 |
| 2 | `verify.py --full` | 实际运行 ref vs kernel + allclose-style 元素级判定（per-dtype rtol+atol，与 main 对齐）| NPU + runtime |

退出码 0 = 全部通过 / 1 = 存在 fail 或 error，结果写入 `verify_results.json`。

### 监控

| 工具 | 数据源 | 用途 |
|------|--------|------|
| `monitor.py` | progress 与 ar_tasks/ 实时 | 批量运行期间另开终端查看队列、phase、heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 至 `dashboard.py` 查看 active task TUI |
| `summarize.py` | 仅 progress JSON | 运行结束后离线汇总，便于复制粘贴 |
| `tail -f batch.log` | claude stdout | 实时查看 hook 输出、Edit、Bash |

### 断点续跑

- `done` 不再重跑
- `error` 默认跳过；通过 `--retry-errored` 重新纳入
- `pending` 自动续跑
- `running`（终止瞬间正在运行的 op）在下次 `run.py` 启动时自动降级为 `error`，写入 note `stale running, demoted on batch restart`，可通过 `--retry-errored` 重新纳入

> 同一 batch 目录不允许并发运行多个 `run.py`（`<batch_dir>/.batch.lock` 排它）。死进程遗留的 stale lock 在下次启动时被判活清理。

### `run.py` 参数

```
batch_dir                       位置参数

--mode {ref-kernel,ref}         整批模式（也接受 manifest.mode；CLI 优先）
--devices N                     必填；透传至每个 op 的 /autoresearch
--worker-url host:port[,...]    经远端 worker 执行 eval

per-op 透传：
  --max-rounds 30
  --eval-timeout 600

batch 兜底：
  --timeout-min 180             单 op wall-clock 上限

队列筛选：
  --only A,B,C
  --limit N
  --retry-errored

调度 / CLI 透传：
  --cooldown-sec 5
  --claude-bin claude
  --model ""
  --extra-claude-arg ...
```

## Skills 库

唯一参考源：`skills/triton/latency-optimizer/references/*.md`。性能调优 markdown（无 frontmatter），按文件名选取：autotune、scalar_to_vector、tiling_optimization、dimension-merge、vector_core_partition、libdevice-usage、load-order、loop-invariant-hoisting、pass-merge、discrete_memory_access、constexpr_parameters、checklist、avoid_scalar_lowering。

PLAN 阶段 hook 提示 Claude Glob 该目录，Read 1-3 个最相关的文件，将文件名写入 plan item rationale。

## 内部机制

外部接口（slash 命令、`task.yaml`、`.ar_state/` 路径）保持稳定。修改内部实现的入口：

| 主题 | 位置 |
|------|------|
| Bash gate（命令在何 phase 合法）| [phase_policy.py](scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [autoresearch/.claude/settings.json](.claude/settings.json)；脚本位于 [hooks/](scripts/hooks/)，命名 `guard_*.py` / `post_*.py` / `stop_*.py` |
| phase 转移 | [phase_machine/state_store.py](scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` 与 `compute_resume_phase` 位于 [phase_policy.py](scripts/phase_machine/phase_policy.py) 末尾 |
| 测时与 verify/profile 实现 | [eval_kernel.py](scripts/engine/eval_kernel.py)（静态脚本，CLI 接 `--phases`）|
| Eval 执行链 | [task_config/eval_client.py](scripts/task_config/eval_client.py) `run_eval` → 本地 [eval_runner.py](scripts/utils/eval_runner.py) `local_eval` → subprocess 调用 [eval_kernel.py](scripts/engine/eval_kernel.py)；远程 [package_builder.py](scripts/task_config/package_builder.py) tar.gz → HTTP POST → [worker/server.py](scripts/worker/server.py) → 相同 `local_eval` |
| Remote worker SSH 调度 | [ar_cli.py](scripts/ar_cli.py) `worker --remote-host` 分支；config 位于 [config.yaml](config.yaml) `remote_worker.hosts` |
| Triton 退化静态检查 | 规则与 AST 实现：[`skills/triton/kernel-verifier/scripts/validate_triton_impl.py`](../skills/triton/kernel-verifier/scripts/validate_triton_impl.py)（canonical）。EDIT 前的运行入口：[`engine/quick_check.py`](scripts/engine/quick_check.py)（其 docstring 列出三类退化模式）。autoresearch/scripts/utils/validate_triton_impl.py 仅为 re-export shim |
| DIAGNOSE 契约 | [CLAUDE.md](CLAUDE.md) 不变量 #9（canonical-form bash）与 #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |

### Hook 接线概要

```
hook_post_edit (Write/Edit kernel.py 之后)
  gate: [ -f autoresearch/.active_task ] || exit 0   ← 依赖 .active_task
  EDIT phase 下编辑 kernel.py → 提示运行 pipeline.py

hook_post_bash (任意 bash 之后)
  检测 "AR_TASK_DIR=" → set_task_dir 写入 .active_task → _fresh_start 设置 BASELINE
  检测 baseline.py → on_baseline_settled (PASS 或任何 FAIL 均进入 PLAN)
  检测 pipeline.py / create_plan.py → 推进对应 phase

hook_guard_bash (bash 之前)
  不依赖 .active_task
  直接读取 .ar_state/.phase 判定允许或禁止
```

`guard_bash` 与 `post_edit` 对 `.active_task` 的依赖不对称：前者不依赖，后者依赖。此为「忘记 export AR_TASK_DIR」症状的成因。
