# AutoResearch

基于 Claude Code 的算子迭代优化框架。Claude 读代码、写 plan、改 kernel、诊断
失败；Hook 负责阶段转移、plan 校验、eval 调度、KEEP/DISCARD、回滚。Eval 默认
走本地 NPU 子进程；可选远端 HTTP worker（dev 机起 orchestrator + 测评机起 daemon，
经 SSH tunnel 透明转发）。

栈固定为 Triton-Ascend kernel + Ascend NPU + PyTorch ref，因此命令面里没有
"选 backend / 选语言"这类参数。

## Quick Start

候选源文件放 [autoresearch/workspace/](../autoresearch/workspace/)，命名
`<op>_ref.py` / `<op>_kernel.py`：

```bash
cd autoresearch    # autoresearch 是自包含子目录, .claude/ 已配置好
claude
```

进入 Claude 后：

```
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 5 --max-rounds 200
```

另开终端 `python scripts/dashboard.py` 看实时进度。

## 启动模式

`/autoresearch` 入参语义：

- `--`-prefixed flag → 新建任务（scaffold + 首次 baseline 原子完成）
- 已存在的目录路径 → resume 该目录
- `--resume` → resume 最近活跃 task
- 无参数 → 交互式询问

`/autoresearch` 只接受一种新建入口：`--ref X.py --kernel Y.py`。
scaffold 自动跑 baseline：seed PASS → phase 直接进 PLAN；seed FAIL →
phase 也直接进 PLAN，第一批 plan items 用于改写 seed kernel。

CLI 用户面只有两个维度：

| flag | 取值 | 说明 |
|------|------|------|
| `--devices` | 本地 NPU 下标，逗号分隔：`5` 或 `0,1,2,3` | **必填** |
| `--no-code-checker` | 关闭静态分析 | 可选 |

`arch`（如 `ascend910b3`）从 `--devices` 选中的卡 `npu-smi info` 推出，写
进 `task.yaml` 仅用于 dashboard / report 显示。

## 主循环

单轮：**PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle**。
连续 3 次 FAIL 切到 DIAGNOSE，plan 全部 settle 切到 REPLAN，预算耗尽切到
FINISH。

```
INIT
  │  /autoresearch --ref X.py --kernel Y.py
  ▼
BASELINE  (scaffold --run-baseline 原子完成，跑 seed kernel)
  │
  ▼
PLAN  (BASELINE PASS 或 FAIL 都进 PLAN；FAIL 时第一批 plan items 改写 seed)
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

DIAGNOSE / REPLAN 不绕回 PLAN——`create_plan.py` 校验通过后 hook 直接写
`phase = EDIT`。每个 plan item 要么在 `history.jsonl` 里有 KEEP / DISCARD /
FAIL 终态，要么在 REPLAN/DIAGNOSE 边界被静默丢弃；pid 单调推进、不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|------|-------------|------|
| BASELINE | `baseline.py` | seed_metric → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）|
| EDIT | Edit kernel.py → `pipeline.py` | history.jsonl + 可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG）|

## Eval 执行

`baseline.py` / `pipeline.py` 直接 in-process 调 `task_config.run_eval`；后者
按 `worker_urls` 是否非空分两条路径：

**本地路径**（默认）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(task_dir, config, device_id)
     └─ utils.eval_runner.local_eval(task_dir, ...)         # 2-pass driver
          ├─ Popen eval_kernel.py --phases profile_base ...  # ref 单独跑
          └─ Popen eval_kernel.py --phases verify,profile_gen ...  # kernel 跑
```

**远程路径**（`--worker-url` 或 `task.yaml worker.urls`）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(..., worker_urls=[...])
     ├─ package_builder.build_package          # tar.gz: task.yaml + ref + kernel + data_files
     └─ HTTP POST /api/v1/run → worker         # multipart
                                  └─ worker/server.py 收包 → safe_extract →
                                     utils.eval_runner.local_eval(同上)
```

两条路径在 worker 内部共用同一份 `local_eval` 实现，绝不会 drift。两次 subprocess
拆分（ref pass 和 kernel pass）是为了 kernel 端的 SIGKILL / 设备 hang 不连累已
落盘的 ref 时延。

`eval_kernel.py` 是静态脚本（不再 codegen），同一进程里串行跑请求的 phase——
verify 之后做 profile_gen，triton JIT 编译缓存自然 hot 给 profile_gen 复用。
调试时可以直接：

```bash
cd <task_dir>
python <repo>/autoresearch/scripts/engine/eval_kernel.py \
    --task-dir . --op-name <op> --kernel-file kernel --ref-file reference \
    --device-id 0 \
    --phases verify,profile_gen
```

verify 失败时 ref 时延仍由 profile_base 单独测得（同进程下一段），与 verify
解耦，dashboard 顶栏始终显示 PyTorch baseline。Sticky baseline 写定后
（`baseline_metric` + `baseline_source=ref` 落 progress.json）后续轮 phase 列表
里直接不带 `profile_base`，省一段。

## 精度容差

verify.py（Tier 2 预检）和 `/autoresearch` 每轮 verify 共用同一份比较函数
[`correctness.py`](../autoresearch/scripts/utils/correctness.py)，对齐
`main` 分支 `skills/triton/kernel-verifier/scripts/verify.py` 的
allclose-style 标准：

- 按 ref dtype 取 `(rtol, atol)`：fp32 → `(2^-13, 1e-5)`，
  fp16 → `(2^-10, 1e-3)`，bf16 → `(2^-7, 1e-2)`，未知 dtype 回落 fp32。
- 元素级判定：`|new - ref| <= atol + rtol·|ref|`，**每个元素都得过**。
- 额外硬性检查：NaN 位置一致、Inf 位置 + 符号一致、bool / int 精确匹配。

## 配置与状态

| 路径 | 用途 |
|------|------|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref / kernel 输入（位于 `autoresearch/workspace/`）|
| `task.yaml` | 任务配置 — name / arch / editable_files / ref_file / devices / eval_timeout / max_rounds / metric / **`data_files`**（sibling JSON/PT/NPZ; scaffold 自动写）/ **`worker.urls`**（远程 worker URL 列表，可选）|
| `.ar_state/.phase` | 当前阶段 |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态）|
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/progress.json` | 运行时状态 |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（CLAUDE.md 不变量 #10）|
| `config.yaml` | `hallucinated_scripts`（脚本名容错映射）+ `remote_worker.hosts`（SSH alias → repo_path / env_script 映射，给 `ar_cli worker --remote-host` 用）|
| `.claude/settings.json` | Hook + 权限配置（已提交进仓库）|
| `.claude/settings.local.json` | API key、model 覆盖（不进 git）|

`.ar_state/` 内除 `plan_items.xml` / `diagnose_v<N>.md`(DIAGNOSE) 外都由
hook 和脚本机控，Claude 不能手写。

## Skills 库

唯一参考源：`skills/triton/latency-optimizer/references/*.md`——
纯 perf 调优 markdown（无 frontmatter），按文件名挑：autotune、
scalar_to_vector、tiling_optimization、dimension-merge、
vector_core_partition、libdevice-usage、load-order、
loop-invariant-hoisting、pass-merge、discrete_memory_access、
constexpr_parameters、checklist、avoid_scalar_lowering。

PLAN 阶段 hook 提示 Claude Glob 这个目录，Read 1-3 个最相关的，把
文件名写进 plan item rationale。

## 内部机制（按需阅读）

外部接口稳定（slash 命令、`task.yaml`、`.ar_state/` 路径）。想动内部时的
入口：

| 想了解 | 看哪里 |
|--------|--------|
| Bash gate（哪条命令在哪个 phase 合法）| [phase_policy.py](../autoresearch/scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [autoresearch/.claude/settings.json](../autoresearch/.claude/settings.json)；脚本在 [hooks/](../autoresearch/scripts/hooks/)，命名 `guard_*.py` / `post_*.py` / `stop_*.py` |
| phase 转移 | [phase_machine/state_store.py](../autoresearch/scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](../autoresearch/scripts/phase_machine/phase_policy.py) 末尾 |
| 测时 + verify/profile 实现 | [eval_kernel.py](../autoresearch/scripts/engine/eval_kernel.py)（静态脚本，CLI 接 `--phases`）|
| Eval 执行链 | [task_config/eval_client.py](../autoresearch/scripts/task_config/eval_client.py) `run_eval` → 本地路径 [eval_runner.py](../autoresearch/scripts/utils/eval_runner.py) `local_eval` → subprocess 调 [eval_kernel.py](../autoresearch/scripts/engine/eval_kernel.py)；远程路径 [package_builder.py](../autoresearch/scripts/task_config/package_builder.py) tar.gz → HTTP POST → [worker/server.py](../autoresearch/scripts/worker/server.py) → 同款 `local_eval` |
| Remote worker SSH 调度 | [ar_cli.py](../autoresearch/scripts/ar_cli.py) `worker --remote-host` 分支；config 在 [config.yaml](../autoresearch/config.yaml) `remote_worker.hosts` |
| Triton 退化静态检查 | [validate_triton_impl.py](../autoresearch/scripts/utils/validate_triton_impl.py)（AST-only）。`engine/quick_check.py` 在 EDIT 之前调它；`skills/triton/kernel-verifier/` 单跑场景也有一份独立副本。 |
| DIAGNOSE 契约 | [CLAUDE.md](../CLAUDE.md) 不变量 #9（canonical-form bash）和 #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](../.claude/agents/ar-diagnosis.md) |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- `torch_npu` + `triton` + CANN（Ascend NPU 上必装）

eval 在哪里跑就在哪里 `cd autoresearch && claude`——卡的可见性靠 `--devices N`
（subprocess 启动时设 `ASCEND_RT_VISIBLE_DEVICES`）。

---

## 远程 Worker（可选）

如果开发机没 NPU、想把 orchestrator 留在本机但 eval 走远端 Ascend 机器，用
`ar_cli worker` 子命令一条命令搞定 SSH 起 daemon + 自动 tunnel：

```bash
# 1) config.yaml 添加 host 信息（一次性）
cat >> config.yaml <<'EOF'
remote_worker:
  hosts:
    my-npu:
      repo_path: /home/<user>/AscendOpGenAgent/autoresearch
      env_script: /home/<user>/env.sh
EOF

# 2) 本机起远程 worker —— ssh 进对端 source env + 起 daemon + 起 ssh -L tunnel
#    用户侧只跑这一条; cleanup 也只一条
python scripts/ar_cli.py worker --remote-host my-npu --start \
    --backend ascend --arch ascend910b3 --devices 6 --port 9111

# 3) 跑 baseline 或 /autoresearch 时点 worker_url
python scripts/engine/baseline.py <task_dir> --worker-url 127.0.0.1:9111
# 或者写进 task.yaml:
#   worker:
#     urls: ["127.0.0.1:9111"]
# 这样 pipeline.py 每轮 EDIT 后自动走远程

# 4) 不用时拆掉
python scripts/ar_cli.py worker --remote-host my-npu --stop --port 9111
```

ar_cli 内部细节：

- `--start --remote-host` 在 dev 端组装 `ssh <alias> 'bash -lc "source env_script && cd repo_path && python scripts/ar_cli.py worker --start --bg ..."'`，远端 daemon log 落 `/tmp/ar_worker_<port>.log`
- 紧接着起 `ssh -f -N -T -L <port>:127.0.0.1:<port> <alias>`，pid 落 `~/.autoresearch_state/tunnels/<port>.pid`（Windows 走 `Get-CimInstance` 抓 pid，POSIX 走 `pgrep`）
- `--stop` 反向：先 `ssh` 远端杀 daemon，再 SIGTERM 本机 tunnel
- `--status` 直接 curl `127.0.0.1:<port>/api/v1/status`（透明走 tunnel）
- 远端 worker `worker/server.py` 只 2 个接口：`/api/v1/run`（multipart：tar 包 + task_id + op_name + timeout）+ `/api/v1/status`；`/run` 收包 → safe_extract → 调用 `local_eval` 同款 2-pass driver → 返回合并后的 verify/profile dict

包装内容（`task_config/package_builder.py`）：`task.yaml` + `ref_file` + `editable_files[*]` + `data_files[*]`。`data_files` 由 scaffold 在创建 task_dir 时按 ref 的 stem 自动扫描 `.json` / `.pt` / `.npz` 兄弟文件并写进 task.yaml（解决 NPUKernelBench 多 shape JSON、sglang 缓存 .pt 等 sibling 依赖在远端找不到的问题）。

---

## 批量跑

`autoresearch/scripts/batch/` 下的脚本对一批 `(ref.py, kernel.py?)` 任务
全自动跑 `/autoresearch`。批级状态（done/error/pending、指向 task_dir 的
链接）落 `<batch_dir>`，round 级状态（plan / history / 各轮 kernel）由
`/autoresearch` 落 `<repo>/ar_tasks/<op>_<ts>_<uuid>/`。

### happy path

约定：cwd 在 `autoresearch/` 自包含子目录；远端长跑用 tmux/screen；
`--devices` 必填，整批串行复用一张卡。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 0. cd 进自包含子目录（.claude/ 已配好；后续路径都相对这里）
cd autoresearch

# 1. 摆 ref/kernel 文件
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. discover + Tier 1 verify（语法 / import / 必备 export）
python scripts/batch/prepare.py $BATCH_DIR

# 3. (可选) Tier 2 预检：本机实跑 ref vs kernel
python scripts/batch/verify.py $BATCH_DIR --full

# 4. 后台跑
# NOTE: tmux 默认起非 login shell，conda activate 会失败。
# 用 bash --login 确保 conda init 生效。
tmux new -d -s ar_batch \
    "bash --login -c 'source YOUR_ENV.sh && \
     python -u scripts/batch/run.py $BATCH_DIR \
       --devices $DEVICE 2>&1 | tee -a $BATCH_DIR/batch.log'"

# 5. 监控（另开终端）
python scripts/batch/monitor.py $BATCH_DIR

# 6. 跑完汇总
python scripts/batch/summarize.py $BATCH_DIR
```

### 启动模式

每个 op 都需要一对 `(ref.py, kernel.py)`。scaffold `--run-baseline` 跑 seed：
- BASELINE PASS → phase 直接 `PLAN` → max-rounds 轮性能优化
- BASELINE FAIL → phase 也直接 `PLAN`，第一批 plan items 改写 seed kernel；
  连续 3 次失败触发 DIAGNOSE

### Batch 目录布局

```
<batch_dir>/                         ← 批级
  manifest.yaml                      # prepare.py 写
  batch_progress.json                # run.py 写：每个 op 的 status / task_dir / metrics
  batch.log                          # run.py 写：claude --print 的全部 stdout
  verify_results.json                # prepare.py / verify.py 写
  refs/<op>_ref.py                   # ⚠️ 文件名必须严格 <op>_ref.py
  kernels/<op>_kernel.py             # ⚠️ 文件名必须严格 <op>_kernel.py

<repo>/ar_tasks/<op>_<ts>_<uuid>/    ← round 级（/autoresearch 自己维护）
  kernel.py / reference.py / task.yaml
  .ar_state/{.phase, progress.json, plan.md, history.jsonl, report.md}
```

`batch_progress.json:cases.<op>.task_dir` 把两层穿起来。

### manifest.yaml

`prepare.py` 自动生成，长这样：

```yaml
ref_dir: refs
kernel_dir: kernels
ops:
- op_a
- op_b
```

加 / 删 ref/kernel 文件后重跑 `prepare.py $BATCH_DIR`，ops 列表整体重扫。

### Verify 两档

| Tier | 触发 | 检查 | 需要硬件 |
|------|------|------|----------|
| 1 | `prepare.py` 默认 | 语法 / import / 必备 export（`Model`, `ModelNew`, `get_inputs`...）| 否 |
| 2 | `verify.py --full` | 实跑 ref vs kernel + allclose-style 元素级判定（per-dtype rtol+atol，与 main 对齐）| NPU + runtime |

退出码 0=全过 / 1=任何 fail/error，结果写 `verify_results.json`。

### 监控

| 工具 | 数据源 | 用途 |
|------|--------|------|
| `monitor.py` | progress + ar_tasks/ 实时 | 跑批时另开终端看队列 / phase / heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 进 `dashboard.py` 看 active task TUI |
| `summarize.py` | 仅 progress JSON | 跑完后离线汇总，复制粘贴友好 |
| `tail -f batch.log` | claude stdout | 看 hook 输出 / Edit / Bash 实时 |

### 断点续跑

总规律：

- `done` 不会重跑
- `error` 默认跳过；`--retry-errored` 捞回
- `pending` 自动续上
- `running`（被杀那瞬间在跑的 op）下次 `run.py` 启动时自动降级为 `error`，
  写 note `stale running, demoted on batch restart`，配 `--retry-errored` 捞回

> ⚠️ 同一 batch 目录不能同时跑两个 `run.py`（`<batch_dir>/.batch.lock`
> 排它）。死进程留下的 stale lock 会在下次启动时被自动判活清理。

### `run.py` 参数

```
batch_dir                       位置参数

--mode {ref-kernel,ref}         整批模式（也接受 manifest.mode；CLI 优先）
--devices N                     必填；透传给每个 op 的 /autoresearch

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

### Autoresearch 内部机制速记（debug 用）

```
hook_post_edit (Write/Edit kernel.py 后)
  └→ gate: [ -f autoresearch/.active_task ] || exit 0   ⚠️ 依赖 .active_task
  └→ EDIT phase 下编辑 kernel.py → 提示跑 pipeline.py

hook_post_bash (任何 bash 之后)
  └→ 检测 "AR_TASK_DIR=" → set_task_dir 写 .active_task → _fresh_start
     直接设 BASELINE
  └→ 检测 baseline.py → on_baseline_settled (PASS / 任何 FAIL 都进 PLAN)
  └→ 检测 pipeline.py / create_plan.py → 推进对应 phase

hook_guard_bash (bash 之前)
  └→ 不依赖 .active_task（关键差异！）
  └→ 直接读 .ar_state/.phase 决定允/禁
```

**关键不对称**：guard_bash 不依赖 `.active_task`，但 post_edit 依赖。这就是
"忘 export AR_TASK_DIR" 症状的成因。
