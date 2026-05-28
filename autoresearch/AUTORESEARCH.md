# AutoResearch

Claude Code 帮你自动优化算子：写 plan → 改 kernel → 测时延 → 判 KEEP/DISCARD
→ 进入下一轮，连续失败自动 DIAGNOSE，预算耗尽自动收尾出报告。
栈固定为 **Triton-Ascend kernel + Ascend NPU + PyTorch ref**。

> 第一次跑？看下面的「**5 分钟教程**」一节，按顺序操作就行。
>
> 已经会用了想看细节？跳到「主循环」「Eval 执行」「远程 worker」「批量跑」。

---

## ⚠️ 命名契约（必须遵守）

这些是 scaffold / batch / package_builder 硬编码的规则，违反一条整个流程就
跑不起来。不是建议、不是惯例 — **是契约**。

### 文件名

| 角色 | 必须叫什么 | 谁强制 |
|------|----------|--------|
| ref Python 文件 | 单跑：任意名，路径通过 `--ref` 传；**批跑：严格 `<op>_ref.py`** | batch/prepare.py 按 stem 配对 |
| kernel Python 文件 | 单跑：任意，路径通过 `--kernel` 传；**批跑：严格 `<op>_kernel.py`** | batch/prepare.py 按 stem 配对 |
| ref/kernel 的 sibling shape/cache 文件 | 文件 stem 必须以 ref stem 为前缀，比如 ref 是 `10_Relu.py` → 配 `10_Relu.json` / `10_Relu_all_case.json` / `10_Relu.pt` 都行；**`2_FloorDivide.json` 不会被认作 sibling** | scaffold.py 按 ref stem 过滤 |
| sibling 后缀 | 只认 `.json` / `.pt` / `.npz`，且文件名不以 `.` 开头 | scaffold.py |

scaffold 把这些 sibling 文件**自动写进** `task.yaml: data_files:`，远端
worker 收 tar 包时按这个列表打包；本地 eval 时 ref 里 `os.path.join(
os.path.dirname(__file__), "<name>.json")` 也能找到。**两条路径都依赖这个
列表**。

### Python 接口

| 角色 | 必须暴露什么 | 谁要 |
|------|----------|------|
| `<op>_ref.py` | 类 `Model` 继承 `nn.Module`；函数 `get_inputs()` **或** `get_input_groups()`（二选一）；函数 `get_init_inputs()` | eval_kernel.py + batch/verify.py Tier-1 |
| `<op>_kernel.py` | 类 `ModelNew` 继承 `nn.Module` | eval_kernel.py + batch/verify.py Tier-1 |

`get_inputs` 单 shape 用，返回 `[t1, t2, ...]`；`get_input_groups` 多
shape 用，返回 `[[t1, t2, ...], [t1', t2', ...], ...]`。**不能同时给** —
`get_input_groups` 优先。

`Model.__init__(*init_inputs)` 和 `Model.forward(*inputs)` 的 arity 必须
跟 `get_init_inputs()` / `get_inputs()[0]` 长度对得上 — eval_kernel 会
直接 `Model(*init_inputs)` 和 `model(*inputs)`，对不上就 INFRA_FAIL。

### task.yaml 字段名

scaffold 自动写，但如果你手改要注意这些字段名是 fixed schema：

```yaml
name:            <str>          # 必填; 决定 task_dir 命名
arch:            <str>          # ascend910b3 等; npu-smi info 推
editable_files:  [kernel.py]    # batch/eval 都按这个找 kernel 文件
ref_file:        reference.py   # scaffold 拷贝时改名成这个
eval:
  timeout:       <int sec>      # per-shape
metric:
  primary:       latency_us     # 字面字符串, eval_assemble 按这个键取
  lower_is_better: true
agent:
  ref_file:      reference.py
  max_rounds:    <int>
devices:         [<int>, ...]
data_files:      [<basename>, ...]   # scaffold 自动写, 你可以手加
worker:                                # 可选; 走远端 eval
  urls:          [<host:port>, ...]
code_checker:
  enabled:       true | false
```

字段名拼错（比如把 `editable_files` 写成 `editable_file`）就被 yaml
loader 默认值兜住，结果你以为生效了其实没生效 — **谨慎手改**。

### task_dir 命名

scaffold 写 `ar_tasks/<op_name>_<int(time.time())>_<uuid.hex[:6]>/`。
`<op_name>` 来自 `--op-name`。这个三段命名是 batch/manifest 识别 task_dir
的依据，**不要手动改 task_dir 的名字**。

---

## 5 分钟教程：跑通你的第一个优化任务

目标：让 Claude 帮你把一个 ReLU 算子从 baseline 的速度做出几倍加速，全程
不用手动改代码。

### 0. 一次性环境检查

| 你需要 | 说明 |
|------|------|
| `cd autoresearch` 后能跑 `claude` | 仓库根的 `autoresearch/` 子目录自包含 `.claude/` 配置，不用 mv |
| `npu-smi info` 能看到至少一张 910B3/910B4 | 用 `npu-smi info` 看 HBM-Usage 找一张空卡，记住编号 |
| 安装好 `torch_npu`、`triton-ascend`、`CANN`、PyYAML | 用 `python -c "import torch_npu"` 验证 |

### 1. 准备 ref（PyTorch 黄金答案）和 kernel（你的种子实现）

> 这里命名其实自由，因为单跑通过 `--ref` / `--kernel` 显式传路径。但
> **如果以后要进批跑，必须严格 `<op>_ref.py` / `<op>_kernel.py`** —
> 详见上面的「命名契约」节。

最小例子放到 `autoresearch/workspace/`（手动创建）：

**`workspace/relu_ref.py`** — 这是 PyTorch 写的标准答案，autoresearch 用它
当 correctness oracle 和 perf baseline：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.float16)]

def get_init_inputs():
    return []
```

**`workspace/relu_kernel.py`** — 你的 triton-ascend 种子实现。先随便写个
最朴素的版本（甚至 wrap PyTorch 也行，只要 Claude 能在它基础上迭代）：

```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_npu:
            x = x.npu()
        return torch.relu(x)
```

> 必须：ref 暴露 `Model` / `get_inputs` / `get_init_inputs`，kernel 暴露
> `ModelNew`。文件命名约定：`<op>_ref.py` / `<op>_kernel.py`。

### 2. 进 Claude，跑 /autoresearch

```bash
cd autoresearch
claude
```

Claude 启动后**在 prompt 里**输入：

```text
/autoresearch --ref workspace/relu_ref.py --kernel workspace/relu_kernel.py \
  --op-name relu --devices 0 --max-rounds 5 --no-code-checker
```

参数解释（够你跑通就够了）：

| flag | 必填 | 含义 |
|------|------|------|
| `--ref` / `--kernel` | ✓ | 上面准备好的两个文件 |
| `--op-name` | ✓ | 起一个标识名，决定 task_dir 命名 |
| `--devices` | ✓ | 用哪张 NPU。`0` / `0,1,2`（逗号分隔） |
| `--max-rounds` | ✗ | 跑几轮就停，第一次推荐 5 轮快速看效果 |
| `--no-code-checker` | ✗ | 关掉 triton 静态退化检查（如果你写的 kernel 不是纯 triton） |

### 3. 你会看到什么

Claude 会按顺序自动执行：

1. **scaffold**：创建 `ar_tasks/relu_<时间戳>_<6 位 hex>/`，把 ref 和
   kernel 拷进去，生成 `task.yaml`，git init 一个隔离的 commit 历史。
2. **BASELINE**：跑 seed kernel 测时延和正确性。stdout 出现一行：
   ```
   [baseline] Initialized: task=relu, seed_latency_us=3.0, baseline(ref)=3.0, commit=...
   ```
3. **PLAN**：Claude 读 ref/kernel，写出 N 条优化思路到 `plan.md`，每条
   一个 `pN` id。
4. **EDIT**：Claude 拿当前 (ACTIVE) 的 plan item，改 `kernel.py`，触发
   `pipeline.py`（hook 自动跑），里面会做：
   - **quick_check**：静态过一下，不让明显错的代码下去测
   - **eval**：双 subprocess 测 ref 时延和 kernel 时延，算 speedup
   - **判 KEEP/DISCARD/FAIL**：变快 → KEEP（git commit 进 task 的隔离仓
     库）；变慢 → DISCARD（revert）；出错 → FAIL（也 revert）
   - **settle**：把 plan item 标 [x]，advance 到下一个
5. **重复 PLAN/EDIT**，连续 3 次 FAIL 自动 DIAGNOSE（subagent 写诊断
   报告，重写 plan），所有 plan 跑完进 REPLAN。
6. **FINISH**：跑满 `max_rounds` 或没改进空间，写
   `.ar_state/report.md`（含一张内嵌 SVG 速度曲线），Claude 收尾停止。

### 4. 另开一个终端实时看进度

```bash
cd autoresearch
python scripts/dashboard.py ar_tasks/relu_<timestamp>_<hex>/ --watch
```

看到 best_metric 一路降、speedup 一路升就是正常优化。

### 5. 跑完之后

报告在 `<task_dir>/.ar_state/report.md`。最佳 kernel 是 `<task_dir>/kernel.py`
（git log 能看到每个 KEEP 的 round 都是一个 commit）。

### 想中途看 / 续跑 / 重来？

| 你想 | 做法 |
|------|------|
| 关 Claude 后续跑 | 重启 Claude 再 `/autoresearch --resume` |
| 续跑指定任务 | `/autoresearch --resume <task_dir>` |
| 不想跑这一轮、想从头再来 | 删掉 `ar_tasks/<task_dir>/`，重新 `/autoresearch --ref ... --kernel ...` |
| 中间想看具体哪轮做了什么 | `<task_dir>/.ar_state/history.jsonl` 一行一轮 |

---

## /autoresearch 命令的完整参数

`/autoresearch` 入参语义：

- `--`-prefixed flag → 新建任务（scaffold + 首次 baseline 原子完成）
- 已存在的目录路径 → resume 该目录
- `--resume` → resume 最近活跃 task
- 无参数 → 交互式询问

`/autoresearch` 只接受一种新建入口：`--ref X.py --kernel Y.py`。
scaffold 自动跑 baseline：seed PASS → phase 直接进 PLAN；seed FAIL →
phase 也直接进 PLAN，第一批 plan items 用于改写 seed kernel。

CLI 用户面：

| flag | 取值 | 说明 |
|------|------|------|
| `--devices` | 本地 NPU 下标，逗号分隔：`5` 或 `0,1,2,3` | **必填** |
| `--max-rounds` | 整数 | 默认 30 |
| `--eval-timeout` | 整数（秒，per shape） | 默认 600 |
| `--no-code-checker` | flag | 关闭 triton 退化静态分析 |
| `--worker-url` | `host:port[,host:port]` | 走远端 worker eval（见「远程 worker」节） |

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
[`correctness.py`](scripts/utils/correctness.py)，对齐
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
| Bash gate（哪条命令在哪个 phase 合法）| [phase_policy.py](scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [autoresearch/.claude/settings.json](.claude/settings.json)；脚本在 [hooks/](scripts/hooks/)，命名 `guard_*.py` / `post_*.py` / `stop_*.py` |
| phase 转移 | [phase_machine/state_store.py](scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](scripts/phase_machine/phase_policy.py) 末尾 |
| 测时 + verify/profile 实现 | [eval_kernel.py](scripts/engine/eval_kernel.py)（静态脚本，CLI 接 `--phases`）|
| Eval 执行链 | [task_config/eval_client.py](scripts/task_config/eval_client.py) `run_eval` → 本地路径 [eval_runner.py](scripts/utils/eval_runner.py) `local_eval` → subprocess 调 [eval_kernel.py](scripts/engine/eval_kernel.py)；远程路径 [package_builder.py](scripts/task_config/package_builder.py) tar.gz → HTTP POST → [worker/server.py](scripts/worker/server.py) → 同款 `local_eval` |
| Remote worker SSH 调度 | [ar_cli.py](scripts/ar_cli.py) `worker --remote-host` 分支；config 在 [config.yaml](config.yaml) `remote_worker.hosts` |
| Triton 退化静态检查 | [validate_triton_impl.py](scripts/utils/validate_triton_impl.py)（AST-only）。`engine/quick_check.py` 在 EDIT 之前调它；`skills/triton/kernel-verifier/` 单跑场景也有一份独立副本。 |
| DIAGNOSE 契约 | [CLAUDE.md](CLAUDE.md) 不变量 #9（canonical-form bash）和 #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |

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
