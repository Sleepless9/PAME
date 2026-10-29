# 基于 gem5 MinorCPU 的 MinorRVV 周期精确建模设计

> 文档定位：本文件是一份可直接用于实现、评审和分阶段开发的微结构设计说明。它描述一个从 gem5 MinorCPU Execute/InstQueue 接收 RVV 指令，并在独立向量后端中按 instruction、beat、lane、operand 和 memory transaction 多级展开的周期精确模型。

## 1. 文档目标与设计边界

### 1.1 建模目标

MinorRVV 的目标不是为每条向量指令指定一个固定 `opLat`，而是让指令延迟由实际资源和流水行为自然产生：

\[
T_{inst}=f(VL,SEW,LMUL,N_{lane},W_{lane},N_{bank},\text{bank conflict},\text{FU latency/II},\text{queue pressure},\text{memory latency})
\]

模型应能够用于以下研究：

- lane 数量、lane 宽度和 sub-lane 并行度的设计空间探索；
- lane-local VRF bank 数量、端口类型、映射函数和冲突率分析；
- operand queue、result queue、旁路和 chaining 的收益分析；
- ALU、MUL、FPU、DIV 等 FU 数量、延迟和启动间隔分析；
- VLSU outstanding、合并策略、memory bandwidth 和 cache 行为分析；
- cross-lane 网络和 slide、gather、reduction 等非局部操作分析；
- instruction-level、beat-level 和 element-group-level 依赖控制分析。

模型至少应满足以下“周期精确”约束：

1. 每个模块有显式容量、延迟、吞吐和背压；
2. VRF bank 端口冲突通过仲裁和排队产生延迟，而不是直接增加经验 penalty；
3. FU 的 latency 与 initiation interval（II）分开建模；
4. memory 指令使用 gem5 timing memory request/response，而不是固定 load/store 延迟；
5. 模块只读取 current state，并在周期边界提交 next state，C++ 调用顺序不能改变硬件时序；
6. squash、drain、fault 和 commit 必须保持 MinorCPU 的精确状态语义。

### 1.2 非目标

第一阶段不要求：

- RTL 信号级或门级精确；
- 完整复刻某个现有 RVV 核的所有结构；
- 一开始就支持激进的乱序向量发射、寄存器重命名或任意 element-level chaining；
- 用一个解析公式代替真实的队列、bank、FU 和 memory 竞争；
- 让 VecBeat 或 LaneBeat 进入 MinorCPU 的 architectural in-flight/commit 结构。

### 1.3 核心粒度

MinorRVV 同时包含五种建模粒度：

| 粒度 | 代表对象 | 主要负责模块 | 生命周期 |
|---|---|---|---|
| Architectural instruction | 一条 `vadd.vv`、`vle32.v` | Minor Execute、Dispatcher、Global Scoreboard | dispatch 到 commit |
| Vector beat | 一段连续 element group | Global Sequencer | 数个周期 |
| Lane beat | 某个 beat 在一个 lane 上的工作 | Lane Sequencer、Lane Scoreboard | 数个周期 |
| Operand/result transaction | 一次 VRF 读写或一次 FU 操作 | Operand Requester、VRF、FU、WB | 1 到多个周期 |
| Memory transaction | 一个 cache-line/fragment 请求 | VLSU | 请求到响应 |

最重要的边界是：

```text
MinorCPU              MinorRVV
───────────────        ─────────────────────────────
MinorDynInst           RvvInstContext
instruction issue  ->  VecBeat -> LaneBeat -> VrfReq
architectural commit <- RvvCompletion
```

MinorCPU 始终把一条 RVV 指令视为一条 `MinorDynInst`；所有 beat、lane 和 bank token 都是 MinorRVV 内部的微结构状态。

## 2. 总体架构

```text
                   Minor Fetch / Decode
                            │
                            ▼
                 Execute Input / InstQueue
                            │
                 vector issue handshake
                            │
                            ▼
                 ┌──────────────────────┐
                 │ RVV Issue Adapter    │
                 │ + Dispatcher         │
                 └──────────┬───────────┘
                            │ VectorInstToken
                            ▼
                 ┌──────────────────────┐
                 │ Inst Context Table   │
                 │ Global Scoreboard    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Global Sequencer     │
                 └──────────┬───────────┘
                            │ VecBeat
           ┌────────────────┼─────────────────┐
           │                │                 │
           ▼                ▼                 ▼
      Lane Array           VLSU       Cross-lane Unit
   ┌────────────────┐   ┌────────┐   ┌──────────────┐
   │ Lane 0 ... N-1 │   │ timing │   │ slide/gather │
   └───────┬────────┘   │ memory │   │ reduction    │
           │            └───┬────┘   └──────┬───────┘
           └────────────────┼───────────────┘
                            ▼
                 Completion / Fault Tracker
                            │ RvvCompletion
                            ▼
                 Minor in-flight / Commit
```

推荐把 MinorRVV 作为 Minor Execute 后的“复合 FU/backend”集成。它在 Minor 的 FU 选择层面可以表现为一个可接受向量指令的执行资源，但内部不能直接继承“整条指令固定延迟”的 `FUPipeline` 语义。

## 3. 与 MinorCPU 的集成方式

### 3.1 复用范围

| MinorCPU 机制 | 处理方式 | 说明 |
|---|---|---|
| Fetch/Decode | 直接复用 | 保留现有取指、预测和解码路径 |
| Execute input/InstQueue | 直接复用 | RVV 指令从这里被选择和握手接收 |
| `MinorDynInstPtr` | 直接复用 | 保留 PC、线程、序号、fault 和 tracing 身份 |
| Scalar scoreboard | 部分复用 | 管理 scalar source/destination 边界依赖 |
| Vector dependency | MinorRVV 自己实现 | 使用 Global/Lane Scoreboard |
| in-flight/commit | 直接复用 | 一条 RVV 指令只占一个 architectural in-flight 项 |
| squash/drain | 扩展 Minor 路径 | Minor 发出控制，RVV 清理内部 token |
| Minor 固定延迟 FU | 不用于整条 RVV | 可复用其自阻塞 pipeline 思想实现 RVV 子 FU |
| Minor LSQ | 不直接承担向量事务 | VLSU 自己拆分事务并连接 timing memory |
| clock/activity | 建议复用 | Minor 和 RVV 初版同频、同 tick |

### 3.2 Issue/accept 握手

Minor Execute 只有在 MinorRVV 真正接受指令时，才能从待发射位置移除该指令并登记为已发射：

```cpp
bool vector_selected = isRvvInst(inst);
bool fire = vector_selected && rvv.canAccept(inst);

if (fire) {
    rvv.acceptNext(inst, scalarSnapshot, vectorConfigSnapshot);
    // 同一个周期边界更新 Minor issue/in-flight 状态
}
```

禁止先 pop 再检查 RVV 是否有空间。`canAccept()` 至少要联合检查：

- Dispatch Queue 是否有空位；
- Context Table 是否有空项；
- Global Scoreboard 是否允许分配；
- 当前线程是否处于 squash/drain/config serialization 状态；
- 指令需要的 scalar operand 是否已由 Minor 准备好；
- 对 P0 中的特殊序列化指令，RVV 后端是否为空。

### 3.3 向量配置快照

`VL`、`VTYPE`、`SEW`、`LMUL`、`vstart`、`vta`、`vma`、`vxrm`、`frm` 等执行所需配置，应在 dispatch/accept 时快照进 `RvvInstContext`。后端执行时不能重新读取“当前 CSR”，否则后续 `vsetvl(i)` 可能错误影响已发射指令。

配置更新必须具备次序语义：

- P0：`vsetvl(i)` 与其他向量指令序列化；
- P1：允许前序向量指令继续执行，但后序指令只能在新配置 architectural 生效后 dispatch；
- 任意实现都必须保证每条指令使用 dispatch 时的配置快照。

### 3.4 Completion 返回

MinorRVV 在最后一个目标 beat 真正写回、最后一个 memory transaction 满足完成策略、且 fault 状态确定后，产生一个整条指令粒度的 `RvvCompletion`。Minor 继续负责程序序 commit。

```text
RVV internal done
      │
      ▼
RvvCompletionQueue
      │ ready/valid
      ▼
Minor Execute completion path
      │
      ▼
program-order commit
```

普通向量算术只返回 done/fault；`vfirst.m`、`vcpop.m`、`vmv.x.s` 等还需要携带 scalar destination/result，并接入 Minor 的 scalar writeback/scoreboard 释放路径。

## 4. 模块划分与建模粒度

| 模块 | 输入粒度 | 输出粒度 | 主要状态/资源 | 建议初始时序 |
|---|---|---|---|---|
| RVV Issue Adapter | `MinorDynInst` | accept/deny | 接口状态 | 组合判断，边界接收 |
| Dispatcher | instruction | `VectorInstToken` | DispatchQ、Context Table | 1 cycle，1 inst/cycle |
| Global Scoreboard | instruction/register | hazard grant | producer/reader 状态 | dispatch 时查询/分配 |
| Global Sequencer | instruction | `VecBeat` | active inst、beat cursor | 1 beat/cycle |
| Lane Sequencer | vector beat | `LaneBeat` | lane-local cursor | 1 lane beat/cycle |
| Lane Scoreboard | lane beat/chunk | ready/grant | operand/dest readiness | 每周期查询和更新 |
| Operand Requester | lane beat | `LogicalVrfReq` | pending operand bitmap | 参数化请求宽度 |
| VRF Mapper | logical request | `VrfBankReq` | 固定映射函数 | 组合或 1 cycle |
| VRF Arbiter/Banks | bank request | read response/write ack | 单端口 bank、仲裁 | read latency 1 cycle |
| Operand Collector/Queue | operand response | `VectorFuReq` | bundle ready bitmap | 队列化、可背压 |
| FU Pipelines | FU request | `VectorFuResp` | pipeline stages | latency/II 分离 |
| ResultQ/WB | FU response | VRF write/complete | result queue、WB arbiter | 1 write/bank/cycle |
| VLSU | memory beat | mem fragment/response | AGU、MSHR、reorder | timing memory |
| Cross-lane Unit | cross-lane beat | per-lane result | network/reduction state | 参数化拓扑/延迟 |
| Completion Tracker | lane/beat done | instruction done | bitmaps/counters/fault | 边界更新 |

## 5. Instruction Context 与全局状态

### 5.1 静态描述和动态状态分离

```cpp
using RvvInstId = uint64_t;
using BeatId = uint32_t;
using LaneId = uint16_t;

struct RvvConfig {
    uint32_t vl;
    uint32_t vstart;
    uint16_t sewBits;
    int8_t lmulLog2;
    bool vta;
    bool vma;
    bool vm;
    uint8_t vxrm;
    uint8_t frm;
};

struct RvvInstDesc {
    RvvInstId id;
    Minor::MinorDynInstPtr minorInst;
    ThreadID tid;
    uint64_t streamSeqNum;
    uint64_t predictionSeqNum;

    VecOpcode opcode;
    VecFuType fuType;
    RegIndex vd;
    RegIndex vs1;
    RegIndex vs2;
    RegIndex vmask;
    bool hasVd;
    bool hasVs1;
    bool hasVs2;
    bool readsOldVd;

    RvvConfig cfg;
    RegVal scalarValue;
    Addr memBase;
};

struct RvvInstState {
    enum class Status {
        Dispatched,
        Waiting,
        Sequencing,
        Executing,
        Completing,
        Done,
        Squashed
    } status;

    uint32_t totalBeats;
    uint32_t issuedBeats;
    uint32_t completedBeats;
    uint32_t outstandingMem;
    Fault fault = NoFault;
    uint32_t faultElement = 0;
    bool killed = false;
};

struct RvvInstContext {
    RvvInstDesc desc;
    RvvInstState state;
};
```

Engine 维护 `InstContextTable<RvvInstId, RvvInstContext>`。模块间 token 只携带 ID、局部 tag 和当前层需要的信息，不能复制整个 context。

### 5.2 Context 生命周期

```text
accept
  │ allocate ID/context/global hazards
  ▼
Dispatched -> Waiting -> Sequencing -> Executing -> Completing -> Done
     │             │           │             │
     └─────────────┴───────────┴──── squash ─┴──> Squashed
```

只有同时满足以下条件才释放 context：

- completion 已被 Minor 接收；或该指令已被 squash；
- 所有携带该 `instId` 的内部 token 已移除或可安全丢弃；
- 所有无法取消的 memory response 已返回并被吸收；
- 不再有 scoreboard producer/reader 状态引用该 context。

## 6. 多级分发与 Sequencer

### 6.1 四级责任边界

```text
Minor InstQueue
    │ 选择“哪条 architectural instruction 发射”
    ▼
Dispatcher
    │ 创建上下文、快照配置、分配依赖
    ▼
Global Sequencer
    │ 把一条长向量指令展开为 VecBeat
    ▼
Lane Sequencer
    │ 把 VecBeat 投影为本 lane 的 LaneBeat 和 operand 访问
    ▼
Lane backend
```

不要让 Global Sequencer 直接生成 bank/row。Global Sequencer 只理解 vector element range；Lane Sequencer 理解 lane 数据宽度和 sub-lane；VRF Mapper 才理解物理 bank topology。

### 6.2 Beat 宽度与数量

定义：

- `NLanes`：物理 lane 数；
- `LaneWidthBits`：每个 lane 的数据通路宽度；
- `PackedSubElements(SEW)`：一个 lane 是否支持多个窄元素并行；
- `ElementsPerLaneBeat = LaneWidthBits / SEW`，仅在支持相应 packed 执行时成立；
- `ElementsPerVecBeat = NLanes * ElementsPerLaneBeat`。

于是：

\[
N_{beat}=\left\lceil\frac{VL-vstart}{ElementsPerVecBeat}\right\rceil
\]

如果 FU 不支持 packed narrow element，则 `ElementsPerLaneBeat` 应取实际执行能力，而不能仅由位宽除法推导。

LMUL 不增加物理 lane。它决定 logical register group、VLMAX 和寄存器地址跨界；在给定 `VL` 后，beat 数主要由有效 element 数和真实执行宽度决定。

### 6.3 Global Sequencer 状态

```cpp
struct SeqState {
    RvvInstId instId;
    uint32_t nextElem;
    BeatId nextBeat;
    uint32_t beatsIssued;
    bool active;
};
```

每周期的发射条件：

```text
active instruction exists
AND target backend route ready
AND output VecBeat channel ready
AND dependency policy permits this beat
AND no serialization/fault stop condition
```

一个推荐的 `VecBeat`：

```cpp
struct BeatTag {
    RvvInstId instId;
    BeatId beatId;
};

struct VecBeat {
    BeatTag tag;
    uint32_t elemBase;
    uint16_t elemCount;
    DynamicBitset activeElements;
    uint32_t groupByteOffset;
    BackendRoute route;
    bool first;
    bool last;
};
```

`activeElements` 是 tail-valid 与 architectural mask 的基础。若 mask 尚未从 v0 读取，可先记录 tail-valid，稍后由 mask requester 合并为实际 element enable。

### 6.4 Lane Sequencer

Lane Sequencer 根据 `VecBeat`、`laneId`、`SEW` 和 lane packed 能力形成 `LaneBeat`：

```cpp
struct LaneTag {
    RvvInstId instId;
    BeatId beatId;
    LaneId laneId;
};

struct LaneBeat {
    LaneTag tag;
    uint32_t firstElem;
    uint8_t elemCount;
    uint32_t groupByteOffset;
    uint64_t elementEnable;
    uint64_t byteEnable;
    bool first;
    bool last;
};
```

Lane Sequencer 负责：

- 计算本 lane 在该 beat 中承载的 element；
- 生成 tail-valid、初始 byte enable 和 mask 请求；
- 根据 opcode 生成 `src0/src1/src2/oldVd/mask` 的 operand 需求；
- 为 lane scoreboard 分配 entry；
- 把逻辑 operand 访问交给 Operand Requester；
- 不直接访问物理 RAM。

## 7. 依赖控制：Global Scoreboard 与 Lane Scoreboard

### 7.1 两层 scoreboard 的区别

```text
Global Scoreboard
  关注：不同 architectural instruction 之间是否存在 RAW/WAR/WAW
  键：architectural vreg 或 register group

Lane Scoreboard
  关注：已放行指令在本 lane 内的 operand/chunk 是否就绪、当前在哪个阶段
  键：LaneTag + local chunk/operand slot
```

Bank conflict 不属于 scoreboard hazard。Bank conflict 由 VRF bank arbiter 处理；scoreboard 只记录依赖和 token 就绪状态。

### 7.2 Global Scoreboard：P0 整寄存器粒度

P0 推荐为每个 architectural vector register 记录：

```cpp
struct GlobalVregState {
    std::optional<RvvInstId> writer;
    uint16_t readerCount;
};
```

对 LMUL register group，dispatch 时展开覆盖的所有 architectural vreg，并执行：

- RAW：任一 source group 有未完成 writer，则等待；
- WAW：任一 destination group 有 writer，则等待；
- WAR：若采用允许多条指令 overlap 的策略，destination group 有 reader 时等待；
- P0 可以用保守的 instruction-complete 释放，先不做 chaining。

mask register、old-vd、scalar source/destination 也必须进入依赖集合。Fractional LMUL 可先保守锁定完整 architectural vreg，后续再细化到 fractional slice。

### 7.3 Lane 专用 Scoreboard

每个 lane 建议独立维护一个容量有限的 entry table：

```cpp
struct LaneScoreEntry {
    LaneTag tag;
    VecOpcode opcode;
    uint8_t requiredOperands;
    std::bitset<4> operandRequested;
    std::bitset<4> operandReady;
    uint16_t pendingVrfReads;
    bool fuIssued;
    bool resultReady;
    bool wbGranted;
    bool complete;
    bool killed;
};
```

状态推进：

```text
Allocated
   │
   ▼
RequestingOperands ── bank conflict/backpressure ──┐
   │                                               │
   ▼                                               │
OperandsReady <────────────────────────────────────┘
   │ FU ready
   ▼
Executing
   │ latency
   ▼
ResultPending
   │ WB bank grant
   ▼
WritebackComplete
```

Lane Scoreboard 使一个 lane 可以同时跟踪多个不同 beat：例如 beat 0 在 FU 中、beat 1 等待 src1、beat 2 正在发 VRF 请求。

### 7.4 P1/P2 chaining 粒度

P1 可把 destination readiness 从“整条 instruction complete”细化为“每 lane 每 beat ready”。producer 完成某个 lane beat 的真实写回或旁路发布后，consumer 对应 lane beat 可以继续。

P2 再进一步细化为 physical chunk/element group readiness：

```cpp
struct ChunkReadyKey {
    RegIndex vreg;
    LaneId lane;
    uint32_t localChunk;
};
```

实现 chaining 时必须明确 ready 的定义：

- `FU result ready`：只允许显式 bypass consumer 使用；
- `VRF write accepted`：允许后续从 VRF 读取；
- `instruction complete`：只用于全局释放和 commit，不应替代细粒度 ready。

## 8. Lane 微结构

```text
                       Vector Lane i
 ┌───────────────────────────────────────────────────────────┐
 │ Lane Sequencer                                            │
 │      │                                                    │
 │ Lane Scoreboard                                           │
 │      │                                                    │
 │ Operand Requesters: src0/src1/src2/oldVd/mask              │
 │      │ LogicalVrfReq                                      │
 │      ▼                                                    │
 │ VRF Address Mapper                                        │
 │      │ VrfBankReq                                         │
 │      ▼                                                    │
 │ Bank Arbiter -> Bank0 ... BankM-1 (single-port RAM)        │
 │      │ VrfReadResp                                        │
 │      ▼                                                    │
 │ Operand Collector / ALUQ / MULQ / FPUQ                     │
 │      │                                                    │
 │      ├──────────┬───────────┬──────────┐                   │
 │      ▼          ▼           ▼          ▼                   │
 │     ALU       MUL/MAC      FPU       optional DIV          │
 │      └──────────┴───────────┴──────────┘                   │
 │                         │                                 │
 │                    Result Queue                           │
 │                         │                                 │
 │              Bypass / Writeback Arbiter                   │
 │                         │                                 │
 │                    VRF Bank Write                         │
 └───────────────────────────────────────────────────────────┘
```

Lane 是一个并行向量数据通道，而不是单个 ALU。每个 lane 可以有多个专用 FU，也可以让昂贵的 DIV/SQRT 被若干 lane 共享。是否共享必须通过显式仲裁和吞吐参数体现。

## 9. Lane-local VRF slice 与多 bank single-port RAM

### 9.1 物理组织

逻辑上 ISA 仍只有 `v0` 到 `v31`。物理上每个 logical vreg 被 striping 到所有 lane；每个 lane 保存整个 VRF 的一个 slice，并继续切成多个 bank：

```text
Logical v7
   │ fixed striping
   ├── Lane0 VRF slice ── B0 B1 ... B7
   ├── Lane1 VRF slice ── B0 B1 ... B7
   ├── Lane2 VRF slice ── B0 B1 ... B7
   └── Lane3 VRF slice ── B0 B1 ... B7
```

推荐初版每个 bank 为：

- 单端口 RAM：每周期最多一次 read 或 write；
- 固定 word width，通常等于 `LaneWidthBits`；
- read latency = 1 cycle；
- write 在周期边界生效；
- 同一 bank 的 read/write、多个 read 或多个 write 均需仲裁。

单端口是推荐配置，不是 RVV ISA 要求。模型应允许以后替换为 1R1W、dual-port 或不同 latency 的 bank。

### 9.2 固定映射与动态结果

物理 topology 和 mapping function 在设计时固定；运行时由 `vreg/groupOffset/SEW/beat` 计算具体 lane、bank、row 和 byte enable。

定义：

- `LaneBytes = LaneWidthBits / 8`；
- `VLENBytes = VLEN / 8`；
- `groupByteOffset` 是相对 LMUL register group 起点的 byte offset；
- `absoluteGroupByte` 先解析为 architectural `vreg` 与寄存器内 byte offset。

对 LMUL ≥ 1：

\[
regOffset=\left\lfloor\frac{groupByteOffset}{VLENBytes}\right\rfloor
\]

\[
physicalVreg=baseVreg+regOffset
\]

\[
byteInVreg=groupByteOffset\bmod VLENBytes
\]

再对固定 lane-sized chunk striping：

\[
globalChunk=\left\lfloor\frac{byteInVreg}{LaneBytes}\right\rfloor
\]

\[
lane=globalChunk\bmod NLanes
\]

\[
laneChunk=\left\lfloor\frac{globalChunk}{NLanes}\right\rfloor
\]

lane 内形成线性 word address：

\[
linearWord=physicalVreg\times ChunksPerVregPerLane+laneChunk
\]

一种简单 bank 映射为：

\[
bank=linearWord\bmod NBanks
\]

\[
row=\left\lfloor\frac{linearWord}{NBanks}\right\rfloor
\]

映射函数必须封装在 `VrfAddressMapper` 中，不能散落在 Sequencer、FU 或 VLSU。这样改变 bank 数量、XOR/hash 映射或 SRAM word width 时，只需替换 mapper。

SEW 只影响 element 解释、sub-lane 选择和 byte enable，不改变底层 bits 的固定摆放。Fractional LMUL 需要额外处理一个 architectural vreg 内的 fractional base/slice；如果 P0 暂不支持，应在 decode/dispatch 明确拒绝或保守处理，不能静默按 LMUL=1 执行。

### 9.3 Logical request 与 Physical request 分层

```cpp
enum class OperandSlot { Src0, Src1, Src2, Mask, OldVd };
enum class VrfAccess { Read, Write };

struct LogicalVrfReq {
    LaneTag tag;
    OperandSlot slot;
    VrfAccess access;
    RegIndex vreg;
    uint32_t groupByteOffset;
    uint16_t sizeBytes;
    uint64_t byteEnable;
    LaneData writeData;
};

struct VrfBankReq {
    LaneTag tag;
    OperandSlot slot;
    VrfAccess access;
    uint16_t bank;
    uint32_t row;
    uint64_t byteEnable;
    LaneData writeData;
};

struct VrfReadResp {
    LaneTag tag;
    OperandSlot slot;
    LaneData data;
    uint64_t validBytes;
};
```

如果一个 logical request 跨两个 RAM word/bank，mapper 必须把它拆成多个 physical request，并由 requester/collector 重新聚合。

### 9.4 Bank 仲裁

每个 bank 每周期收集来自以下来源的请求：

- ALU/MUL/FPU operand read；
- mask/old-vd read；
- FU result writeback；
- VLSU load response writeback；
- cross-lane result writeback；
- 可选的 bypass spill 或调试访问。

P0 推荐优先级：

```text
oldest/fault-critical writeback
  > load response writeback
  > regular FU writeback
  > operand read
```

同一优先级采用 round-robin，避免 starvation。未获 grant 的 request 保持 valid，不能丢失或重新生成重复事务。所有冲突延迟都由请求等待自然产生。

## 10. Operand Requester、Collector 与 Queue

### 10.1 Operand Requester

每个 LaneBeat 根据 opcode 形成 operand requirement：

| 指令类型 | 典型 operand |
|---|---|
| `vadd.vv` | vs1、vs2、可选 mask |
| `vadd.vx` | vs2、scalar snapshot、可选 mask |
| `vfmacc.vv` | vs1、vs2、old vd、可选 mask |
| masked undisturbed | source、old vd、mask |
| vector store | vector source、address/index、mask |
| reduction | source、initial accumulator、mask |

Requester 为每个 slot 只产生一次逻辑请求，并在 scoreboard 中标记 `requested`。请求因 channel/bank 背压未 fire 时必须保持稳定。

### 10.2 Operand Collector

VRF responses 可能乱序返回，因此通过 `LaneTag + OperandSlot` 聚合：

```cpp
struct OperandBundle {
    LaneTag tag;
    VecOpcode opcode;
    std::array<LaneData, 4> src;
    std::bitset<4> required;
    std::bitset<4> ready;
    uint64_t elementEnable;

    bool allReady() const { return (ready & required) == required; }
};
```

Collector 只在所有 required operand ready 且目标 FU input ready 时发出 `VectorFuReq`。不同 FU 建议使用独立 operand queue，避免一个被阻塞的长延迟 FU 阻塞无关 ALU 指令。

### 10.3 Queue 的作用

Operand queue 用于吸收：

- VRF bank conflict 导致的 operand 到达抖动；
- FU input backpressure；
- 不同 operand latency；
- 多个 beat 的重叠执行。

队列容量必须参数化并统计 full/occupancy/stall。禁止把无限容器当成时序队列，否则性能结果会失真。

## 11. FU Pipelines

### 11.1 请求与响应

```cpp
struct VectorFuReq {
    LaneTag tag;
    VecOpcode opcode;
    uint16_t sewBits;
    std::array<LaneData, 4> src;
    uint64_t elementEnable;
    uint8_t frm;
    uint8_t vxrm;
};

struct VectorFuResp {
    LaneTag tag;
    LaneData data;
    uint64_t byteEnable;
    uint8_t fpFlags;
    bool saturated;
};
```

FU 内部可以复用 Minor `SelfStallingPipeline<T>` 的思想，但 payload 必须是 beat/lane 粒度，不能仍是整条 `QueuedInst`。

### 11.2 Latency 与 II

每个 FU 至少配置：

- `latency`：输入 fire 到结果可用的周期数；
- `initiationInterval`：连续两次接受新请求的最小间隔；
- `inputQueueDepth`、`outputQueueDepth`；
- `numUnitsPerLane` 或共享范围；
- 支持的 SEW/packed 模式。

例：FMA `latency=4, II=1` 表示四级流水但每周期可接收一个新 LaneBeat；不能错误实现为每四周期接收一个。

### 11.3 Mask、tail 和 undisturbed 语义

实际写回 byte enable 至少由以下条件组合：

```text
physical byte belongs to active element
AND element index < VL
AND element index >= vstart
AND (vm || mask bit)
AND instruction-specific enable
```

对 mask/tail undisturbed，需要读取并合并 old vd，或者使用精确 byte write enable 保留未写字节。对 agnostic policy，模型可以写入定义的实现值或保留旧值，但必须在功能模型和验证中采用一致策略。

## 12. Result Queue、Bypass 与 Writeback

FU 完成不等于 beat 完成：

```text
FU result ready
    │
    ▼
Result Queue
    │ wait for target bank
    ├──── optional bypass publish
    ▼
VRF write grant
    │
    ▼
write committed at cycle boundary
    │
    ▼
LaneComplete
```

Result Queue entry 至少包含 `LaneTag`、逻辑 destination、group byte offset、data、byte enable 和 exception flags。写回仍走 Logical VRF Write → Mapper → Bank Arbiter，读写共享同一物理映射。

P0 不做 bypass，消费者等 producer 全指令完成。P1 可加入同 lane 同 chunk 的 producer-to-consumer bypass；P2 再考虑多个 FU、VLSU 和 cross-lane 间的 forwarding network。旁路不能绕过依赖跟踪，发布和消费都必须更新 lane scoreboard。

## 13. VLSU

### 13.1 模块结构

```text
VecBeat / LaneBeat
       │
       ▼
Address Generation
       │
element accesses
       ▼
Fragmenter / Coalescer
       │
memory transactions
       ▼
Request Queue -> sendTimingReq()
       │              │ retry
       │              └──────────┐
       ▼                         │
gem5 Memory System              │
       │ recvTimingResp()        │
       ▼                         │
Outstanding Table / Reorder <───┘
       │
       ├── load result -> lane VRF WB
       └── store completion/fault
```

### 13.2 数据结构

```cpp
using MemTxnId = uint64_t;

struct VectorMemFragment {
    MemTxnId txnId;
    LaneTag tag;
    Addr addr;
    uint16_t size;
    bool isLoad;
    bool isStore;
    LaneData storeData;
    uint64_t byteEnable;
    RegIndex vd;
    uint32_t vdGroupByteOffset;
    uint32_t firstElement;
};

struct OutstandingMem {
    MemTxnId txnId;
    LaneTag tag;
    RegIndex vd;
    uint32_t vdGroupByteOffset;
    uint16_t size;
    uint32_t firstElement;
    bool killed;
};
```

gem5 `PacketPtr` 只应存在于 VLSU 内部。用 `txnId` 将 timing response 恢复为 RVV tag，不能把 `PacketPtr` 传播到 lane/FU 通用数据通路。

### 13.3 访问类型与阶段建议

- P0：unit-stride load/store，按 lane/word 拆分，有限 outstanding，不做复杂合并；
- P1：cache-line coalescing、strided access、load response reorder；
- P2：indexed、segment、fault-only-first、复杂 misalignment 和更细粒度 memory ordering。

### 13.4 Store 与精确状态

已发送到 memory system 的 store 通常无法 squash。P0 推荐采用保守策略：

- 未确认不会被旧分支 squash 前，不向 memory system 发出不可逆向量 store；或
- 先写入 RVV store buffer，在指令到达可提交点后释放到 memory system。

load 被 squash 后，已发出的 response 仍必须被端口接收并从 outstanding table 移除，但数据不得写回 VRF。

### 13.5 Fault 与 `vstart`

VLSU 应记录最早 fault element，并停止产生更晚的 architectural side effect。Completion 携带 fault 与 fault element；精确异常路径用该 element 更新 `vstart`。P0 可以序列化 vector memory 指令来降低实现难度，但不能把 fault 简化为无 element 位置的普通固定延迟错误。

## 14. Cross-lane 单元

以下操作不应强行通过“任意 VRF bank 到任意 lane 的全连接”完成：

- `vslideup/vslidedown`；
- `vrgather`；
- `vcompress`；
- reduction；
- 某些 mask/permute 操作。

推荐增加独立 `CrossLaneUnit`：

```text
lane-local source collection
          │
          ▼
CrossLane Request Queue
          │
     topology model
 ring / tree / staged crossbar
          │
          ▼
per-lane result fragments
          │
          ▼
normal lane ResultQ / VRF WB
```

P0 可将 cross-lane 指令序列化并使用固定但显式的分阶段延迟；P1 建模 slide/reduction；P2 再加入 gather/compress 的 bank 与网络竞争。即使 P0 使用简化模型，也必须通过 ready/valid、有限容量和 completion token 与主流水交互。

## 15. Completion、Commit 与资源释放

### 15.1 分层完成 token

```cpp
struct LaneComplete { LaneTag tag; };
struct BeatComplete { BeatTag tag; };

struct RvvCompletion {
    RvvInstId instId;
    Minor::MinorDynInstPtr minorInst;
    Fault fault;
    bool hasScalarResult;
    RegIndex scalarDest;
    RegVal scalarResult;
};
```

```text
all active LaneComplete for a beat
                │
                ▼
          BeatComplete
                │
all required beats/memory effects complete
                │
                ▼
          RvvCompletion
                │
                ▼
          Minor commit
```

一个被 mask/tail 完全关闭的 lane 仍需产生逻辑完成，或者在创建 beat tracker 时被标记为“不需要该 lane”，否则会永久等待。

### 15.2 完成定义

- 算术目的寄存器指令：所有有效 destination 写回已在周期边界提交；
- 无目的寄存器指令：所有 required side effect 已完成；
- load：所有 transaction 返回并完成 VRF 写回，或 fault 已精确确定；
- store：根据配置选择“进入不可撤销 store buffer”或“memory response 完成”；P0 推荐后者；
- reduction/scalar result：最终 scalar result 已可交给 Minor writeback；
- killed instruction：所有不可取消外部 response 已被吸收，且不会再产生 architectural side effect。

## 16. Squash 与 Drain

### 16.1 Squash tag

所有 context 保留 `tid`、`streamSeqNum` 和 `predictionSeqNum`。Minor 发出 squash 时：

1. Context Table 将匹配的 younger instruction 标记为 `killed`；
2. DispatchQ、Sequencer、lane scoreboard、operand/result queue 移除相关 token；
3. FU pipeline 中不能立即移除的 token在输出时丢弃；
4. VLSU outstanding load response 返回后吸收并丢弃；
5. 未释放的 store 必须从 store buffer 取消；已产生不可逆 side effect 的 store 说明模型违反了投机约束；
6. 释放或回滚对应 scoreboard reader/writer 状态；
7. 不向 Minor 发送正常 completion。

`engine.isKilled(instId)` 应是所有长生命周期模块可查询的统一接口。

### 16.2 Drain 条件

`rvv.isDrained()` 只有在以下全部成立时返回 true：

- DispatchQ 和待分发 channel 为空；
- Context Table 无活动项；
- Global Sequencer idle；
- 所有 lane scoreboard、operand queue、FU pipeline、result queue 为空；
- 所有 VRF read pipeline 无 pending response；
- VLSU request/retry/outstanding/reorder/store buffer 为空；
- CrossLaneUnit idle；
- CompletionQ 为空；
- 没有 scheduled RVV internal event 会再次激活状态。

Drain 期间停止接受新指令，但继续 tick 直到已有工作完成。checkpoint/切换前不能只检查 Minor 原有 in-flight 和 LSQ。

## 17. 模块间 token 与 ready/valid 接口

### 17.1 小 token、大状态

推荐原则：模块之间传“小 token”，模块内部维护“大状态”。

```text
Global: RvvInstContext  <- instId lookup

Dispatcher -> Sequencer: VectorInstToken
Sequencer -> Lane:       VecBeat
Lane -> VRF mapper:      LogicalVrfReq
Mapper -> bank:          VrfBankReq
Bank -> collector:       VrfReadResp
Collector -> FU:         VectorFuReq
FU -> ResultQ:           VectorFuResp
VLSU internal:           VectorMemFragment
Completion:              Lane/Beat/RvvCompletion
```

禁止设计一个包含 instruction、bank、operands、result、PacketPtr 和 completion 的全局“大一统 `RvvPacket`”。

### 17.2 Channel 语义

```cpp
template<class T>
class RvvChannel {
  public:
    bool valid() const;
    const T &bits() const;
    bool ready() const;
    bool fire() const { return valid() && ready(); }

    void pushNext(const T &value);
    void popNext();
    void advance();
};
```

约束：

- producer 在 `valid && !ready` 时必须保持 token 稳定；
- 只有 `fire` 才发生一次传输；
- `pushNext` 在本周期结束时进入 channel，consumer 最早下周期看到；
- queue 满时 ready 拉低，背压沿链路返回；
- 每个 channel 有明确 depth；depth=1 表示流水寄存器，不等同无限 queue；
- broadcast 到多 lane 时，要么所有 required consumer 同时 ready 才 fire，要么显式保存 per-lane accepted bitmap。

### 17.3 推荐 channel 列表

| Channel | 建议深度 P0 | 说明 |
|---|---:|---|
| Minor → Dispatch | 1 | accept 边界 |
| DispatchQ | 4 | instruction-level 解耦 |
| Sequencer → Lane fanout | 1~2 | VecBeat 广播/投影 |
| LaneSeq → Operand Requester | 2~4 | lane scoreboard entry 对应 |
| Operand Requester → bank | 每 slot 1 | 未 grant 保持 |
| Bank read response | bank latency pipeline | 不能无界 |
| Operand queue/FU | 2~4/FU | 吸收 bank 抖动 |
| FU result queue | 2~4/FU | 吸收 WB 冲突 |
| CompletionQ | 2~4 | 与 Minor commit 解耦 |

## 18. Current/Next State 周期语义

### 18.1 基本规则

所有模块在 cycle N：

1. 只读取 `current` state 和 current channel；
2. 计算 ready、grant 和 fire；
3. 写入 `next` state/channel；
4. 到周期边界统一 `advance/commitNextState()`；
5. cycle N+1 才能观察新状态。

```cpp
void RvvEngine::evaluate()
{
    // 以下顺序不能改变逻辑结果；各模块只写 next state
    completion.evaluate();
    writeback.evaluate();
    for (auto &lane : lanes)
        lane.evaluate();
    crossLane.evaluate();
    vlsu.evaluate();
    globalSequencer.evaluate();
    dispatcher.evaluate();

    commitNextState();
}
```

如果交换上述函数顺序会改变时序，说明 current/next 隔离不完整。

### 18.2 明确延迟定义

本文统一采用：请求在 cycle N 的边界 fire，latency=1 的响应在 cycle N+1 可见。

例：VRF 1-cycle read：

```text
cycle N:   VrfBankReq fire
cycle N+1: VrfReadResp valid
```

例：FU latency=3、II=1：

```text
cycle N:     req0 fire
cycle N+1:   req1 fire
cycle N+2:   req2 fire
cycle N+3:   resp0 valid
cycle N+4:   resp1 valid
```

若 output queue 满，self-stalling pipeline 是否整体冻结或只阻塞最后一级必须在 FU 模型中明确，不能由容器实现偶然决定。

### 18.3 示例时序

假设：4 lanes、64 bit/lane、SEW=32、支持每 lane 两个 32-bit 元素、VL=16，因此共 2 个 VecBeat；无 bank conflict；VRF read=1、ALU latency=1、WB=1。

```text
Cycle 0  Minor/RVV accept instruction
Cycle 1  Dispatch complete; context visible
Cycle 2  Global Sequencer emits Beat0
Cycle 3  LaneSeq allocates Beat0; emits Beat1
Cycle 4  Beat0 VRF reads fire; LaneSeq allocates Beat1
Cycle 5  Beat0 operands return; Beat1 VRF reads fire
Cycle 6  Beat0 ALU fire; Beat1 operands return
Cycle 7  Beat0 result enters ResultQ; Beat1 ALU fire
Cycle 8  Beat0 WB fire; Beat1 result enters ResultQ
Cycle 9  Beat0 LaneComplete visible; Beat1 WB fire
Cycle 10 Beat1 LaneComplete/InstructionComplete visible
Cycle 11 RvvCompletion visible to Minor
```

具体实现可减少或增加显式 pipeline stage，但必须由参数和 channel 定义，而不是隐藏在 C++ 直接调用中。

## 19. P0 / P1 / P2 分阶段实现

### 19.1 P0：可验证的保守周期模型

目标：先获得正确、稳定、可统计、具备真实资源冲突的模型。

- 与 Minor 同频，由 Execute tick 驱动；
- 1 条 instruction/cycle dispatch，有限 DispatchQ；
- instruction-level Global Scoreboard，RAW/WAR/WAW 整寄存器阻塞；
- Global Sequencer 每周期最多 1 个 VecBeat；
- 每 lane 一个 Lane Sequencer 和 Lane Scoreboard；
- lane-local VRF slice，多 bank single-port RAM，read latency=1；
- 显式 Operand Requester、Collector、OperandQ；
- ALU/MUL/FPU latency 与 II 参数化；
- 显式 ResultQ、WB bank arbitration；
- 不做 chaining/bypass；
- VLSU 支持 unit-stride timing request，有限 outstanding；
- cross-lane 指令先序列化或明确标记 unsupported；
- conservative vector store；
- 完整 squash、drain、completion；
- 支持 mask/tail 和 LMUL ≥ 1；fractional LMUL 如未实现必须显式检查。

P0 完成标准：所有延迟能由 trace 解释；改变 lane/bank/FU 参数会产生符合资源瓶颈预期的变化；随机 squash/drain 不泄漏 context 或 outstanding transaction。

### 19.2 P1：beat-level overlap 与 memory 能力

- Global Scoreboard 增加 beat/lane readiness；
- 同 lane beat-level chaining；
- FU result bypass；
- 多条无依赖向量指令在不同 FU/阶段 overlap；
- cache-line coalescing、strided memory、load response reorder；
- slide 和 tree reduction；
- fractional LMUL 完整映射；
- 更精确 mask/old-vd traffic；
- round-robin/age-based bank 与 WB 仲裁；
- 可配置独立 CrossLane queue。

#### 19.2.1 当前 P1 实现边界

当前实现已落地 beat/lane readiness、同映射 RAW chaining、FU result
bypass、fractional LMUL 低位 slice、mask/old-vd operand traffic、可配置
VRF bank 仲裁、cache-line coalescing、strided address、有限 outstanding 与
确定性乱序响应，以及独立的 slide/tree-reduction CrossLane queue。WAW、WAR
和部分重叠寄存器组仍按 instruction 粒度保守阻塞。

当前 fixed-stripe mapper 要求 `VLEN` 是 `NLanes * LaneWidthBits` 的整数倍，
使每个 architectural vector register 都在 lane 0 重新开始映射。默认拓扑与
RISC-V ISA 一致，为 `VLEN=256, NLanes=4, LaneWidthBits=64`；若 ISA 使用
`VLEN=128`，应将 RVV engine 配置为例如 2 个 64-bit lane。

`VL/VTYPE/SEW/LMUL/vta/vma/vm` 来自指令解码快照；`vstart/vxrm/frm`
在 RVV engine 接受 `InstQueuePacket` 时从对应 `ThreadContext` 快照，此后各
模块只读取 `RvvInstContext`。

本阶段 `VectorMemoryUnit` 是微结构 transaction/coalescing/reorder 的
timing-only 模型，不创建 gem5 `PacketPtr`，也不重复执行 architectural
load/store side effect。RVV timing 完成后只调用 `MinorInstQueue::complete()`；
Minor Execute 在顺序 commit 路径中把真实访存交给 Minor LSQ，并由 Minor
负责 retire。因而当前 synthetic outstanding 可在 host cancel 时直接移除；
接入真实 timing port 后，应改为吸收已经发出的 killed response。indexed、
segment、fault-only-first 和精确 memory fault/vstart restart 仍属于 P2。

### 19.3 P2：高性能和复杂指令

- physical chunk/element-group chaining；
- indexed/segment/fault-only-first memory；
- 更激进的 multiple-issue Global Sequencer；
- 多 FU/共享 FU/异构 lane 配置；
- gather/compress 与参数化 cross-lane topology；
- register cache、更多 bypass、load-to-use forwarding；
- 更细粒度 speculative store buffer 和 precise restart；
- 多线程、公平性、独立 RVV clock domain；
- 可选 vector register renaming/乱序实验。

每个阶段都应在前一阶段的 token 和 channel 契约上扩展，避免重写架构。

## 20. 推荐类层级与文件组织

```text
src/computing_cluster/RVV/
├── rvv_types.hh
├── rvv_channel.hh
├── rvv_engine.hh/.cc
├── rvv_dispatcher.hh/.cc
├── rvv_context.hh/.cc
├── global_scoreboard.hh/.cc
├── global_sequencer.hh/.cc
├── vector_lane.hh/.cc
├── lane_sequencer.hh/.cc
├── lane_scoreboard.hh/.cc
├── operand_requester.hh/.cc
├── operand_queue.hh/.cc
├── vector_reg_file.hh/.cc
├── vrf_mapper.hh/.cc
├── vrf_bank.hh/.cc
├── vector_fu.hh/.cc
├── vector_alu.hh/.cc
├── vector_mul.hh/.cc
├── vector_fpu.hh/.cc
├── result_queue.hh/.cc
├── vector_lsu.hh/.cc
├── cross_lane_unit.hh/.cc
└── completion_tracker.hh/.cc
```

推荐类关系：

```text
Minor::Execute
└── RvvEngine
    ├── RvvDispatcher
    ├── InstContextTable
    ├── GlobalVectorScoreboard
    ├── GlobalSequencer
    ├── VectorLane[NLanes]
    │   ├── LaneSequencer
    │   ├── LaneScoreboard
    │   ├── OperandRequester[]
    │   ├── VectorRegFile
    │   │   ├── VrfAddressMapper
    │   │   ├── VrfBankArbiter
    │   │   └── VrfBank[NBanks]
    │   ├── OperandQueue[FuType]
    │   ├── VectorFuPipeline[FuType]
    │   ├── ResultQueue[FuType]
    │   └── WritebackArbiter
    ├── VectorLSU
    ├── CrossLaneUnit
    └── CompletionTracker
```

`RvvEngine` 对 Minor 暴露的最小接口：

```cpp
class RvvEngine {
  public:
    bool canAccept(const Minor::MinorDynInstPtr &) const;
    void acceptNext(const Minor::MinorDynInstPtr &,
                    const ScalarSnapshot &,
                    const RvvConfig &);
    void evaluate();
    bool hasCompletion() const;
    const RvvCompletion &completion() const;
    void consumeCompletionNext();
    void squash(ThreadID, uint64_t streamSeqNum,
                uint64_t predictionSeqNum);
    bool isDrained() const;
    void drain();
};
```

## 21. 推荐参数

以下默认值是 P0 起点，不是 ISA 要求：

| 参数 | 推荐 P0 默认值 | 说明 |
|---|---:|---|
| `VLEN` | 256 或 512 bit | architectural vector length |
| `NLanes` | 4 | 物理 lane 数 |
| `LaneWidthBits` | 64 | 每 lane datapath/VRF word 宽度 |
| `VrfBanksPerLane` | 8 | lane-local bank 数 |
| `VrfBankPorts` | 1 shared R/W | 单端口 |
| `VrfReadLatency` | 1 cycle | 请求到响应 |
| `VrfMapperLatency` | 0 或 1 | 必须明确配置 |
| `DispatchQueueDepth` | 4 | instruction queue |
| `MaxInflightRvvInsts` | 4~8 | Context Table 容量 |
| `GlobalSeqIssueWidth` | 1 beat/cycle | P0 |
| `LaneScoreboardEntries` | 4~8/lane | lane 内在途 beat |
| `OperandQueueDepth` | 2~4/FU/lane | 按 FU 独立 |
| `ResultQueueDepth` | 2~4/FU/lane | 吸收 WB 冲突 |
| `ALULatency/II` | 1/1 | 初值 |
| `MULLatency/II` | 2~3/1 | 参数化 |
| `FMALatency/II` | 4/1 | 参数化 |
| `DivLatency/II` | variable/>1 | 单独模型 |
| `VlsuRequestQueueDepth` | 8~16 | memory fragments |
| `VlsuMaxOutstanding` | 8~16 | timing requests |
| `CompletionQueueDepth` | 4 | 与 Minor 解耦 |
| `CrossLaneTopology` | serialized/P0 | 后续 tree/ring/crossbar |
| `ChainingGranularity` | instruction | P1 改 beat，P2 改 chunk |

必须把 architectural 参数、微结构容量、pipeline latency 和仲裁策略分开配置，便于 DSE：

```text
Architectural: VLEN, ELEN, supported SEW/LMUL
Topology: lanes, lane width, banks, FU count
Capacity: queue depth, scoreboard entries, outstanding
Timing: mapper/VRF/FU/network latency and II
Policy: arbitration, chaining, store completion, coalescing
```

## 22. 统计、Trace 与不变量

### 22.1 必要统计项

- 每类 RVV 指令数、平均/分位 latency；
- dispatch/global-sequencer stall 原因；
- 每 lane active cycles 和有效 element 利用率；
- 每 bank read/write 次数、conflict 次数、仲裁等待周期；
- operand/result queue occupancy 和 full stall；
- 每 FU issue、busy、output-blocked、利用率；
- chaining/bypass 命中；
- VLSU requests、bytes、coalescing ratio、retry、outstanding occupancy；
- cross-lane traffic 和 stall；
- squash 清理 token/transaction 数；
- 每条指令从 accept 到 completion 的阶段时间戳。

### 22.2 推荐 trace 格式

每条记录至少包含：

```text
cycle, instId, beatId, laneId, module, event, resource, reason
```

示例：

```text
120,17,2,1,VRF,STALL,Bank3,writeback_priority
121,17,2,1,VRF,READ_FIRE,Bank3,src1
122,17,2,1,OPQ,OPERAND_READY,src1,-
123,17,2,1,ALU,ISSUE,ALU0,-
124,17,2,1,WB,WRITE_FIRE,Bank5,vd
```

### 22.3 运行时断言

至少加入以下 invariant：

- 一个 token 的 `instId` 必须存在，或已被明确标记为 killed-awaiting-response；
- 同一 single-port bank 每周期最多一个 request fire；
- 同一个 operand slot 不能被重复接收；
- 未满足 required operand 的 bundle 不能进入 FU；
- FU II 约束不能被违反；
- 一个 LaneTag 只能产生一次 LaneComplete；
- `completedBeats <= issuedBeats <= totalBeats`；
- killed instruction 不得写 architectural VRF/scalar state；
- completion 前 outstanding memory 必须满足完成策略；
- drain 返回 true 时所有内部容器和 scheduled work 均为空。

## 23. 验证计划

### 23.1 功能与时序分层验证

1. 功能结果与参考 ISA 模型比较；
2. 独立验证 address mapper：随机 `vreg/SEW/LMUL/offset` 的 lane/bank/row/byte enable；
3. 独立验证 single-port bank 仲裁不会丢请求或重复服务；
4. 对小 VL 指令逐周期检查预期时间线；
5. 随机插入 queue full、FU backpressure、memory retry；
6. 随机 branch squash 和 drain；
7. 比较统计不变量和 token 守恒。

### 23.2 最小测试矩阵

| 类别 | 必测场景 |
|---|---|
| Element-wise | VL=0、1、刚好一 beat、多 beat、最后半 beat |
| SEW | 8/16/32/64，支持和不支持 packed 的路径 |
| LMUL | m1/m2/m4/m8、寄存器组跨界、非法对齐、fractional |
| Mask/tail | masked、tail、vma/vta 的四类组合 |
| Dependency | RAW/WAR/WAW、old-vd、mask dependency |
| Bank | 无冲突、读读冲突、读写冲突、WB starvation 压力 |
| FU | latency/II、多个 beat、output queue backpressure |
| Memory | unit stride、跨 cache line、misaligned、retry、fault |
| Control | squash before issue/in FU/after load request、drain |
| Cross-lane | slide 边界、reduction 非整 beat、gather 冲突 |

### 23.3 性能 sanity check

无冲突、pipeline 已填满时，算术指令应近似：

\[
T \approx L_{front}+L_{operand}+L_{FU}+L_{WB}+(N_{beat}-1)\times II_{bottleneck}
\]

增加 lane 数应先降低 beat 数，直到 VRF/FU/WB/memory 成为瓶颈；增加 bank 数只应在 bank conflict 占主要瓶颈时显著收益；把 FMA latency 从 4 改 8、II 保持 1 时，长向量稳态吞吐不应减半，但短向量首结果和总 latency 会增加。这些关系可作为模型正确性的性能检查。

## 24. 推荐实现顺序

1. 定义 `rvv_types.hh`、tag、context 和有限 ready/valid channel；
2. 接通 Minor accept/completion/squash/drain 空壳；
3. 实现 Dispatcher、Context Table 和 instruction-level Global Scoreboard；
4. 实现 Global Sequencer 与 Lane Sequencer，只生成 trace，不执行计算；
5. 实现 VRF mapper、single-port banks 和仲裁单元测试；
6. 接入 Operand Requester/Collector/Queue；
7. 实现 ALU pipeline、ResultQ、WB 和 completion；
8. 扩展 MUL/FPU、mask/tail/old-vd；
9. 接入 VLSU timing memory 和 fault；
10. 实现 cross-lane P0 路径；
11. 完成 squash/drain 压力测试和统计；
12. 在 P0 稳定后添加 beat-level chaining 和 coalescing。

## 25. 设计结论

MinorRVV 最合适的架构边界是：

```text
MinorCPU 管：
  architectural instruction、scalar 依赖、in-flight、精确 commit、全局控制

MinorRVV 管：
  vector context、beat/lane 展开、VRF/FU/memory 资源竞争、内部完成
```

其核心流水为：

```text
Minor InstQueue
  -> Dispatcher
  -> Global Scoreboard
  -> Global Sequencer
  -> Lane Sequencer
  -> Lane Scoreboard
  -> Operand Requester
  -> lane-local banked VRF
  -> Operand Queue
  -> FU Pipeline
  -> Result Queue / WB
  -> Lane/Beat/Instruction Completion
  -> Minor Commit
```

这一结构保留了 MinorCPU 的成熟前端和 commit 语义，又允许 RVV 延迟由 lane、VRF bank、queue、FU、cross-lane 和 memory 的真实竞争自然涌现。通过 instruction → beat → lane → transaction 的分层 token、固定的 logical-to-physical VRF mapper、两级 scoreboard，以及严格的 current/next 时序语义，模型可以从 P0 的保守实现平滑演进到支持 chaining、复杂 VLSU 和高性能 cross-lane 网络的 P1/P2，而无需推翻基础接口。
