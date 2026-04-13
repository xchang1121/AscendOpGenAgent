# AscendOpGenAgent 贡献规范

---

## 一、PR 合入规范

### 1.1 PR 标题格式

```
[<scope>] <描述>
```

| scope | 适用场景 |
|-------|---------|
| `triton` | Triton Ascend 侧改动 |
| `ascendc` | AscendC 侧改动 |
| `benchmark` | Benchmark case / 评测逻辑 |
| `router` | op-router / 路由逻辑 |
| `infra` | CI、脚本、构建 |
| `docs` | 文档 |

示例：
- `[triton] 新增 layernorm 算子生成支持`
- `[ascendc] dsl-lowering tiling pass 优化`
- `[benchmark] NPUKernelBench level2 新增 10 case`
- `[router] op-router 增加 CUDA→Ascend 路由分支`

### 1.2 合入门禁

所有 PR 合入前须通过双通路冒烟和 Benchmark 评测两项门禁，确保主干代码质量不退化。

#### 1.2.1 双通路冒烟测试

**所有非文档 PR** 须分别验证 Triton / AscendC 两条主流程，端到端跑通（编译通过 + 精度正确）。

**执行方式**：使用当前仓库中的主 Agent 执行评测：

- **Triton 通路**：调用 `triton-ascend-coder` 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确
- **AscendC 通路**：调用 `ascend-kernel-developer` 生成一个基础算子（如 level1 problem1），验证编译通过 + 精度正确

两条通路都通过后在 PR 模板中标记结果。若因环境原因无法运行，须在 PR 中说明。

> **退化豁免**：PR 中写明原因 + follow-up plan，2 个 maintainer Approve 后可豁免。
> 文档 / 基础设施类 PR 仅需不影响代码功能，无需上述门禁。



#### 1.2.2 Benchmark 评测

跑完 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) 全部任务，精度性能不劣化。

| 门禁 | 含义 | 算子生成 | 性能优化 | 其他 |
|------|------|:---:|:---:|:---:|
| 通过率不退化 | 编译/精度通过数 >= BASELINE | 必须 | 必须 | - |
| Speedup 不退化 | 平均值及逐任务 >= BASELINE × 0.95 | 必须 | 必须 | - |
| 性能有提升 | 至少 1 个算子加速比提升 >= 5% | - | 必须 | - |

> "其他"指 Benchmark、框架改动、Bug 修复、文档等类型的 PR。

**执行方式**：使用仓库自带的 benchmark 脚本执行评测：

```text
bash utils/run_benchmark_triton.sh --benchmark-dir <KernelBench路径> --level <level> --ids <任务列表> --output <输出目录>
bash utils/run_benchmark_ascendc.sh --benchmark-dir <NPUKernelBench路径> --level <level> --ids <任务列表> --output <输出目录>
```

评测完成后，将输出目录中的报告与 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) 逐项对比，在 PR 模板中填写结果。


### 1.3 基线管理

维护 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md)，记录 main 分支最新评测结果。

**更新规则**：算子生成/性能优化 PR 合入 main 后，若性能提升须更新 BASELINE.md（日期、数据、通过率、平均 Speedup）。

### 1.4 Review 规则

| PR 类型 | 最少 Approve | 特殊要求 |
|---------|------------|---------|
| 算子生成 / 性能优化 | 2 | 至少 1 个对应 DSL 侧 maintainer |
| 框架改动（跨侧） | 2 | 两侧各 1 个 maintainer |
| 其他 | 1 | - |

**Review 重点**：
- 算子生成：生成逻辑正确性、prompt 质量、reference 准确性、性能数据真实性
- 性能优化：优化手段合理性、性能数据可复现性
- Benchmark：case 代表性、baseline 合理性、评测公平性
- 框架改动：格式一致性、路由正确性、向后兼容

**性能数据真实性**：Reviewer 有权要求在指定设备上重跑。提交者必须记录测试环境（设备型号、CANN 版本、PyTorch 版本）。

---

## 二、新功能接入流程

1. **创建 Agent**：`agents/<name>.md`
2. **创建 Skill**：`skills/<skill-name>/`
3. **接入主流程或子 Agent 编排**：根据功能归属，更新对应主 Agent 的工作流与文档说明
4. **添加 Benchmark Case**：至少 5 个 case
5. **提供基线数据**：跑一轮 benchmark，追加到 [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md)
6. **提交 PR**：按第一节规范填写模板
