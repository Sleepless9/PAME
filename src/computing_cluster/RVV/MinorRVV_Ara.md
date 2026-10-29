# 基于 gem5 MinorCPU 的 MinorRVV 周期精确建模设计（Ara 周期行为对齐版）

> 本文是一份可直接用于开发、代码评审和周期验证的实现说明。目标是在保留 gem5 MinorCPU 前端、发射、squash、drain 和精确 commit 语义的前提下，使 RVV 后端的模块边界、资源冲突和周期推进尽可能贴近 Ara RVV 1.0 微架构。

> Ara 公开 RTL 是微结构与时序基线；MinorCPU 控制协议和 gem5 timing memory 是仿真适配层。除非显式标记为“研究扩展”，实现不应增加 Ara 中不存在的通用旁路、无限队列、零延迟模块或更激进的乱序能力。

## 1. 对齐目标与关键决策

### 1.1 对齐范围

必须对齐的关键行为：

- Dispatcher 与 main/global sequencer 按整条 vector instruction 工作；
- global sequencer 分配 vector instruction ID、维护 running bitmap 和 global hazard table；
- global sequencer 将 instruction-level PE request 广播到 lanes、VLSU、SLDU 和 MASKU；
- lane sequencer 将一条 PE request 转换为多个 operand requester command 和 VFU command；
- 每个 operand requester 使用地址、剩余长度、EEW 和 hazard metadata 逐个 64-bit VRF word 推进；
- chaining 由 producer 的 VRF write progress 驱动，不等待整条指令完成；
- 每 lane 保存一个 VRF slice，每个 slice 使用 8 个 64-bit 单端口 RAM banks；
- VRF read latency 为 1 cycle，同一 bank 每周期最多一次读或写；
- operand queues 位于 VRF 与 VFU 之间，并显式传播背压；
- ALU、MFPU、load、slide、mask 和 store operand 访问共同竞争 VRF banks；
- memory natural packing 与 VRF lane organization 之间存在 shuffle/deshuffle；
- VRF 保存 EEW tag，需要时由 Dispatcher 注入 reshuffle micro-op；
- reduction 使用 intra-lane reduction 与 SLDU inter-lane exchange，而不是抽象成理想树；
- FU 完成、VRF 写回、vector instruction 完成和 Minor commit 是四个不同事件。

### 1.2 不要求对齐的内容

- SystemVerilog 信号名和组合逻辑逐行复刻；
- 物理综合时钟树、门控时钟和布线延迟；
- CVA6 特有的前端和 AXI 接口实现；
- Ara 代码中的 bug 或非架构性实现偶然行为；
- Ara 之外的通用寄存器重命名、乱序 vector issue 和全局 FU-to-FU bypass。

### 1.3 最重要的建模修正

旧模型以 `Global Sequencer -> VecBeat -> LaneBeat` 为主要执行链。Ara 对齐版改为：

```text
Minor InstQueue
    ↓ architectural instruction
Dispatcher
    ↓ VectorInstToken
Global Sequencer
    ↓ PeRequest：整条指令级广播
Lane Sequencer
    ├── OperandRequestCmd[src0/src1/src2/mask/old-vd/...]
    └── VfuOperation
Operand Requester
    ↓ 每次一个 lane-local 64-bit VRF word
VRF / Operand Queue / VFU / Writeback
```

可以保留 `BeatTag` 作为 trace 和统计标签，但不得为它增加 Ara 中不存在的固定一级 pipeline latency，也不得由全局 sequencer 预先展开整条指令的所有 beat。

## 2. 总体架构

```text
                       MinorCPU
                Fetch / Decode / Execute
                           │
                    InstQueue issue
                           │ ready/valid
                           ▼
               ┌────────────────────────┐
               │ RVV Issue Adapter      │
               │ Dispatcher             │
               │ vector CSR / reshuffle │
               └────────────┬───────────┘
                            │ VectorInstToken
                            ▼
               ┌────────────────────────┐
               │ Global Sequencer       │
               │ ID allocator           │
               │ running bitmap         │
               │ global hazard table    │
               │ target VFU counters    │
               └────────────┬───────────┘
                            │ PeRequest broadcast
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
  Lane Array              VLSU              SLDU / MASKU
  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
  │ LaneSeq      │   │ AddrGen      │   │ all-lane ports  │
  │ Requesters   │   │ VLDU / VSTU  │   │ slide/reshuffle │
  │ 8-bank VRF   │   │ shuffle      │   │ gather/compress │
  │ Operand Qs   │   │ timing mem   │   │ mask/reduction  │
  │ ALU / MFPU   │   └──────┬───────┘   └────────┬────────┘
  └──────┬───────┘          │                    │
         └──────────────────┼────────────────────┘
                            ▼
                Instruction Completion Tracker
                            │ RvvCompletion
                            ▼
                 Minor in-flight / Commit
```

Ara lanes、VLSU、SLDU 和 MASKU 是并列目标单元。global sequencer 按 `targetVfus` 判断一条指令需要哪些单元，并且只有所需目标均能接收时才完成本次广播。

## 3. 与 MinorCPU 的集成

### 3.1 责任边界

| MinorCPU 负责 | MinorRVV 负责 |
|---|---|
| fetch、decode、branch prediction | vector decode 补充信息和 target VFU 选择 |
| scalar register dependency | vector RAW/WAR/WAW 和 chaining |
| `MinorDynInst`、PC、thread、seqNum | Ara vector instruction ID 和 running bitmap |
| architectural in-flight 和 program-order commit | lane word、VRF、VFU、VLSU 内部执行 |
| branch squash、interrupt、drain 请求 | 内部 token 清理和不可取消 response 吸收 |
| scalar architectural state | vector architectural state 和 vector exception metadata |

一条 RVV 指令在 Minor 中始终只对应一个 `MinorDynInst`。Ara instruction ID、operand command、VRF word 和 memory burst不得进入 Minor in-flight queue。

### 3.2 Issue handshake

```cpp
bool selected = isRvvInst(inst);
bool fire = selected && rvv.canAccept(inst);

if (fire) {
    rvv.acceptNext(inst, scalarSnapshot, vectorConfigSnapshot);
    minorMarkIssued(inst);
}
```

`canAccept()` 必须联合检查：

- Dispatcher input register/queue；
- Context Table 空项；
- Ara instruction ID 是否可分配；
- vector configuration serialization；
- scalar source 是否 ready；
- reshuffle/segment micro-op 是否占用 dispatcher；
- drain、squash 和 exception 状态。

只有 `fire` 后 Minor 才能移除待发射指令。RVV 满时必须对 Minor 反压。

### 3.3 配置快照

dispatch 时保存：

```text
VL, VSTART, VTYPE, SEW, LMUL, VTA, VMA, VM, VXRM, FRM
scalar operand, memory base/stride, instruction EEW
```

后端不得读取之后被 `vsetvl(i)` 改写的 CSR。配置类指令与后续 vector instruction 之间必须保持 architectural 次序。

### 3.4 Completion 返回

```cpp
struct RvvCompletion {
    RvvInstId contextId;
    AraInsnId araId;
    Minor::MinorDynInstPtr minorInst;
    Fault fault;
    uint32_t faultElement;
    bool hasScalarResult;
    RegIndex scalarDest;
    RegVal scalarResult;
};
```

RVV completion 只表示指令已具备退休条件；最终 architectural commit 仍由 Minor 完成。

## 4. 周期精确语义

### 4.1 Current/next state

所有模块每周期：

1. 只读取 current state；
2. 计算 ready、valid、grant 和 fire；
3. 只写 next state；
4. 周期边界统一提交 next state；
5. 新状态最早在下一周期可见。

如果调换 C++ `evaluate()` 调用顺序会改变结果，说明 current/next 隔离不完整。

### 4.2 Ready/valid

```cpp
fire = valid && ready;
```

- `valid && !ready` 时，producer 必须保持 payload 稳定；
- 只有 fire 才发生一次传输；
- 接收方本周期写入 next state，下周期才能消费；
- 每个 channel、FIFO、spill register 都必须有有限容量；
- 广播请求必须保存 per-target accepted bitmap，防止某些单元重复接收。

### 4.3 Latency 与 throughput

- VRF request 在 cycle N fire，1-cycle read response 在 N+1 valid；
- VRF write grant 在 N fire，写入和 `resultWritten` 在 N+1 可见；
- FU `latency` 与 `initiationInterval` 独立；
- output backpressure 必须按所选 pipeline 模型冻结末级或全 pipeline；
- memory 延迟由 gem5 timing path、retry 和 response 产生。

## 5. 全局上下文与 Token

### 5.1 Instruction context

```cpp
using RvvInstId = uint64_t;   // gem5 context ID
using AraInsnId = uint8_t;    // 有限 Ara-style in-flight ID

struct RvvInstContext {
    RvvInstId contextId;
    AraInsnId araId;
    Minor::MinorDynInstPtr minorInst;
    ThreadID tid;
    uint64_t streamSeqNum;
    uint64_t predictionSeqNum;

    VecOpcode opcode;
    TargetVfuMask targetVfus;
    RvvConfig cfg;
    OperandDesc operands;

    bool dispatched;
    bool peRequestAccepted;
    TargetVfuMask targetDone;
    Fault fault;
    bool killed;
};
```

`contextId` 用于 gem5 生命周期和 squash；`araId` 用于模拟 Ara 有限 instruction slots、hazard matrix、running bitmap 和 result-written 信号。

### 5.2 Instruction-level request

```cpp
struct PeRequest {
    RvvInstId contextId;
    AraInsnId araId;
    VecOpcode opcode;
    TargetVfuMask targetVfus;

    RegIndex vd;
    RegIndex vs1;
    RegIndex vs2;
    bool useVdAsOperand;
    bool vm;

    uint32_t vl;
    uint32_t vstart;
    VType vtype;
    EewInfo eew;
    RegVal scalarOperand;
};
```

`PeRequest` 是 global sequencer 的主要输出。它是一条整 vector instruction，不是一个执行 beat。

### 5.3 Lane-local tag

```cpp
struct LaneWordTag {
    RvvInstId contextId;
    AraInsnId araId;
    LaneId lane;
    OperandSlot slot;
    uint32_t wordIndex;
};
```

`wordIndex` 是该 operand requester 已发出的 64-bit lane-local word 序号。trace 可以从它推导 element range。

## 6. Dispatcher

Dispatcher 负责：

- 完整 RVV decode；
- vector CSR state 和 dispatch snapshot；
- 合法性、LMUL 对齐、EEW 和 register group 检查；
- 计算 target VFU mask；
- 标量 operand 转发；
- 根据 VRF EEW tag 判断是否需要 reshuffle；
- 把 reshuffle 或 segment memory 指令拆成内部 micro-op；
- 将最终 instruction token 送给 global sequencer；
- 保存原 `MinorDynInst` 与所有内部 micro-op 的父子关系。

### 6.1 Reshuffle 插入

当 source/destination 当前 EEW layout 与本次操作要求不匹配，且 tail/mask policy 需要保留未覆盖字节时：

```text
original instruction
      │
      ├── optional RESHUFFLE(vs/vd)
      │
      └── real operation
```

reshuffle 走 SLDU，并访问所有 lanes。原指令只有在内部 reshuffle 与实际操作都完成后才向 Minor 返回 completion。

### 6.2 Segment memory micro-op

若支持 segment load/store，Dispatcher/SegmentSequencer 将其拆成若干非 segment memory micro-op。P1 为 Ara 对齐阶段，必须显式建模这些 micro-op 的串行/流水关系，不能把整个 segment 指令设为固定延迟。

## 7. Global Sequencer

### 7.1 状态

```cpp
struct GlobalSequencerState {
    DynamicBitset freeIds;
    DynamicBitset runningIds;
    std::array<ReadWriteSet, NrVInsn> regSets;
    HazardMatrix globalHazards;
    std::array<uint16_t, NumTargetVfus> vfuInflight;
    std::optional<PendingBroadcast> pending;
};
```

### 7.2 Hazard matrix

对新指令 N 与每条 running 指令 P，计算：

```text
RAW: N.readSet  intersects P.writeSet
WAR: N.writeSet intersects P.readSet
WAW: N.writeSet intersects P.writeSet
```

Global Sequencer 不需要等所有 hazard 消失才广播。它可以携带每个 operand 的 producer hazard bitmap，让 lane operand requester 在真正读对应 VRF word 时决定是否 stall。这一点是 Ara chaining 的关键。

```cpp
struct OperandHazards {
    DynamicBitset src0;
    DynamicBitset src1;
    DynamicBitset src2;
    DynamicBitset oldVd;
    DynamicBitset mask;
};
```

### 7.3 广播与目标接受

```cpp
struct PendingBroadcast {
    PeRequest req;
    TargetVfuMask required;
    TargetVfuMask accepted;
};
```

一条 request 可以被不同目标在不同周期接受，但已接受目标不得重复 sample。只有 `accepted == required` 时 global sequencer 才能处理下一条输入指令。

### 7.4 VFU counters 与 gold-ticket 语义

为 ALU、MFPU、VLSU、SLDU、MASKU 维护有限 inflight counter。已被计入资源但因广播 backpressure 停留的 pending instruction 必须保留资格，不能被后来者抢占导致永久饥饿。

### 7.5 Completion

各目标单元产生 `targetDone[araId]`。只有全部 required target 完成，global sequencer 才：

- 清除 `runningIds[araId]`；
- 清理 hazard matrix 的对应行列；
- 释放 Ara ID；
- 通知 CompletionTracker。

## 8. Lane Sequencer

Lane Sequencer 接收整条 `PeRequest`，而不是预先展开的 VecBeat。它负责：

- 防止重复 sample 同一 Ara ID；
- 根据 lane ID、VL、VSTART、VSEW、EEW 计算本 lane 的开始元素和有效长度；
- 为每个 operand queue 生成独立 `OperandRequestCmd`；
- 产生一个 instruction-level `VfuOperation`；
- 处理 widening/narrowing、reduction、slide、mask、store 和 indexed-address operand；
- 在 exception flush 时撤销相关 requester command；
- 对 `vrgather/vcompress` 等特殊操作执行额外状态机协调。

```cpp
struct OperandRequestCmd {
    AraInsnId araId;
    RegIndex vs;
    OperandSlot slot;
    uint32_t laneVl;
    uint32_t laneVstart;
    Eew eew;
    ResizeMode resize;
    TargetFu targetFu;
    DynamicBitset producerHazards;
    bool scaleVl;
    bool isReduction;
    bool isSlide;
};

struct VfuOperation {
    AraInsnId araId;
    VecOpcode opcode;
    RegIndex vd;
    uint32_t vl;
    uint32_t vstart;
    VType vtype;
    RegVal scalarOperand;
    bool useScalar;
    bool useOldVd;
    bool vm;
};
```

## 9. Operand Requester 与 Ara 式 Chaining

### 9.1 每个 requester 的 metadata

每个 lane 至少为以下 slots 提供独立 requester：

```text
AluA, AluB,
MulFpuA, MulFpuB, MulFpuC,
MaskB, MaskM,
StoreA,
SlideAddrGenA
```

每个 requester 维护：

```cpp
struct RequesterMetadata {
    AraInsnId araId;
    VrfWordAddress nextAddress;
    uint32_t remainingElements;
    Eew eew;
    DynamicBitset hazards;
    bool widening;
    DynamicBitset wawToggle;
    RequesterState state;
};
```

状态机：

```text
IDLE
  │ accept OperandRequestCmd and OpQueueCmd
  ▼
REQUESTING
  │ each cycle:
  │ check hazard progress
  │ check target OpQueue ready
  │ select current bank
  │ request one 64-bit word
  │ on grant: addr++, remaining -= elementsPerWord
  ▼
IDLE when remaining == 0
```

### 9.2 `resultWritten` 事件

每 lane 为每个 Ara ID 维护 producer write progress：

```cpp
struct ResultWriteProgress {
    uint64_t wordsWritten;
    bool pulseThisCycle;
    bool instructionWriteDone;
};
```

只有 producer 的结果真正获得目标 VRF bank write grant，并在周期边界提交后，才产生 `resultWritten[producerId]`。FU result ready 不能直接解除普通 RAW/WAW hazard。

### 9.3 Chaining 规则

dependent requester 可以按照 producer 的写回速率逐 word 前进：

```text
Producer word 0 write committed
        ↓ resultWritten pulse/progress=1
Consumer word 0 VRF read allowed

Producer word 1 write committed
        ↓ progress=2
Consumer word 1 VRF read allowed
```

建议实现为计数关系，而不是只有单周期 pulse：

```cpp
bool canReadWord(const RequesterMetadata &r, uint32_t consumerWord)
{
    for (AraInsnId p : r.hazards)
        if (producerProgress[p].wordsWritten <= consumerWord)
            return false;
    return true;
}
```

对 widening WAW，producer 每个 source word 可能产生两个 destination writes，需要使用 toggle/counter 调整消费者推进比率。narrowing、conversion 和不同 EEW 也必须使用真实 write-count 比率。

### 9.4 与“lane scoreboard”的关系

Ara 对齐实现不再设置一个统一、理想化的 Lane Scoreboard 来直接判定所有 beat ready。原文中的 Lane Scoreboard 改为三个分布式状态：

```text
Lane Sequencer        ：instruction/VFU command 状态
Operand Requester     ：operand address、remaining、hazard、read progress
Result Progress Table ：每个 Ara ID 的真实 VRF write progress
```

可以保留 `LaneProgressTracker` 做调试和断言，但它只能镜像上述真实状态，不能额外放宽或阻塞时序。

## 10. Ara 风格 Lane-local VRF

### 10.1 固定基线

```text
LaneWidthBits       = 64
VrfBanksPerLane     = 8
VrfBankPorts        = 1 shared read/write
VrfReadLatency      = 1 cycle
VrfWriteVisibility  = next cycle
```

每个 logical vector register 横跨所有 lanes。每个 lane 保存该寄存器的一个 slice；slice 再条带化到 8 个 banks。

```text
Logical VRF
   ├── Lane0 slice -> B0 B1 B2 B3 B4 B5 B6 B7
   ├── Lane1 slice -> B0 B1 B2 B3 B4 B5 B6 B7
   └── LaneN slice -> B0 B1 B2 B3 B4 B5 B6 B7
```

### 10.2 Lane Organization

Ara VRF 不应只实现与 SEW 无关的连续 lane-sized chunk 布局。需要实现 Lane Organization：同一 vector 的连续 elements 分配到连续 lanes，每个 lane 内按 64-bit word 和 element width 解释 narrow elements。

至少提供以下纯函数：

```cpp
LaneId laneForElement(uint32_t elemIndex, Eew eew);
uint32_t laneWordForElement(RegIndex vreg, uint32_t elemIndex, Eew eew);
uint8_t byteOffsetInWord(uint32_t elemIndex, Eew eew);
BankId bankForLaneWord(uint32_t laneWord);
RowId rowForLaneWord(uint32_t laneWord);
```

Ara 基线的 bank/row 拆分：

```text
bank = laneWordAddress low log2(8) bits
row  = laneWordAddress >> log2(8)
```

### 10.3 EEW tag

每个 architectural vector register或可独立追踪的 register slice 保存当前物理 layout 的 EEW tag：

```cpp
std::array<EewTag, 32> vrfEewTag;
```

load、完整覆盖写和 reshuffle 完成时更新 tag。部分写、mask/tail undisturbed 和 widening/narrowing 时必须根据实际覆盖范围决定是否需要提前 reshuffle。

### 10.4 Memory shuffle/deshuffle

```text
Memory Natural Packing
      │ loadShuffle(SEW)
      ▼
VRF Lane Organization

VRF Lane Organization
      │ storeDeshuffle(EEW tag)
      ▼
Memory Natural Packing
```

shuffle/deshuffle 必须消耗有限数据通路和 queue bandwidth，不可作为零延迟数组重排。

### 10.5 VRF request

```cpp
struct VrfWordReq {
    LaneWordTag tag;
    VrfAccess access;
    BankId bank;
    RowId row;
    LaneData64 writeData;
    uint8_t byteEnable;
    OperandQueueId targetQueue;
};

struct VrfReadResp {
    LaneWordTag tag;
    OperandQueueId targetQueue;
    LaneData64 data;
};
```

## 11. VRF Bank 仲裁

每个 lane 的每个 bank 每周期只允许一个 request fire。请求来源分为两级：

```text
High priority group:
  ALU/MFPU operand reads
  ALU/MFPU writeback
  MASKU local requests

Low priority group:
  SLDU/AddrGen operand
  VLDU result writeback
  store/index/memory mask operand reads
```

组内使用 round-robin；组间默认 high priority 覆盖 low priority，以复现 Ara 基线。若研究公平仲裁，必须作为独立 policy 配置，并标记为非 Ara baseline。

未获 grant 的 requester 保持 current address、remaining 和 valid，不得增加地址或重复扣减元素。写回的 final grant 只有在数据真实提交到 VRF 后产生，用于 hazard release 和 target completion。

## 12. Operand Queues 与 VFU

### 12.1 Operand queues

每个 lane 建议使用与 Ara 对应的独立 queues：

```text
ALU A / ALU B
MFPU A / MFPU B / MFPU C
Mask B / Mask M
Store A
Slide/AddrGen A
```

每个 queue 接收 instruction-level command 和随后的 64-bit word stream。它必须：

- 记录该 command 的 EEW、element count、conversion 和 target FU；
- 按 word 对齐多个 source operands；
- 对上游 VRF requester 提供 ready；
- 对下游 FU 提供 valid；
- 在 exception/squash 时按 Ara ID flush；
- 使用有限深度。

### 12.2 FU 组织

每 lane：

- 一个 64-bit SIMD ALU datapath；
- 一个 MFPU cluster，包含 FP pipeline、SIMD multiplier 和 multi-cycle divider；
- narrow SEW 在 64-bit word 内进行 packed SIMD；
- reduction 允许把 partial result送到 SLDU；
- ALU/MFPU 的 result 分别进入各自 writeback stream register/result queue。

```cpp
struct VfuWordReq {
    AraInsnId araId;
    uint32_t wordIndex;
    VecOpcode opcode;
    Eew eew;
    LaneData64 srcA;
    LaneData64 srcB;
    LaneData64 srcC;
    uint8_t byteEnable;
};
```

### 12.3 FU latency 与 II

参数必须分别描述：

```text
latency             ：request fire 到 result valid
initiationInterval  ：连续 request fire 的最小周期距离
outputBlockingMode  ：末级阻塞或全流水冻结
```

Ara baseline 参数应从目标 RTL 配置或测得 trace 固定；探索参数另设 profile，不能无标记混用。

## 13. Writeback 与普通 Chaining

```text
FU result valid
      │
      ▼
result stream register / queue
      │ wait for bank arbiter
      ▼
VRF write grant
      │
      ▼
cycle boundary commits write
      │
      ├── producerProgress.wordsWritten++
      ├── resultWritten pulse
      └── possible final grant
```

普通 producer-consumer chaining 必须经过 VRF write progress。默认禁止：

```text
ALU output ──direct──> next ALU/MFPU input
```

除非 Ara 对该特殊路径确有直接 forwarding，例如 reduction partial result。若增加通用 bypass，应命名为 `ExperimentalDirectBypass`，默认关闭，并单独统计其收益。

## 14. VLSU

### 14.1 结构

```text
PeRequest
   │
   ▼
AddrGen
 unit / strided / indexed / segment
 MMU translation / exception
   │ memory burst descriptors
   ├───────────────┐
   ▼               ▼
VLDU              VSTU
   │               ▲
timing responses   │ lane StoreA operands
   │               │
Load Result Queue  │
   │ shuffle       │ deshuffle
   ▼               │
per-lane VRF WB    │
```

### 14.2 gem5 timing memory 适配

Ara 使用 AXI burst；gem5 中用 `Packet` 和 timing request模拟。需要保留一个 `AraBurstDesc` 层，使 cache-line 拆分不会改变 Ara AddrGen/VLDU/VSTU 的 issue/commit counters：

```cpp
struct AraBurstDesc {
    MemTxnId txnId;
    AraInsnId araId;
    Addr addr;
    uint32_t bytes;
    uint16_t beats;
    bool incrementing;
    bool load;
    bool store;
    uint32_t firstElement;
};
```

一个 Ara burst 可映射为一个或多个 gem5 packets。VLSU completion 以 Ara burst/byte counters、result queue 和 final VRF grants 为准，不能以最后一个 Packet 返回时间简单代替。

### 14.3 P1 必须支持的 memory 行为

- unit-stride、strided、indexed load/store；
- segment memory micro-op；
- mask operand 与 indexed-address operand从 lane OpQ 进入 AddrGen/VSTU；
- load result queue；
- natural-packing 到 lane-organization shuffle；
- store deshuffle；
- misalignment 与 cache-line/packet split；
- MMU/timing translation 或明确的等价延迟接口；
- page/access/misaligned exception；
- fault-only-first；
- `vstart` 和部分 load result drain；
- request retry、outstanding transaction 和 response 吸收。

### 14.4 Store 与 squash

Minor 可能 squash 年轻指令，而外部 store 不可撤销。必须采用：

```text
VSTU builds buffered store bursts
      │
      ▼
Minor confirms non-speculative/releasable point
      │
      ▼
sendTimingReq
```

已发出的 load response即使所属指令 killed 也必须接收并清除 outstanding，但不得写 VRF。

## 15. SLDU、MASKU 与 Reduction

### 15.1 SLDU

SLDU 是连接所有 lanes 的独立单元，负责：

- slide up/down；
- power-of-two 或支持范围内的 stride permutation；
- VRF reshuffle；
- reduction partial result 的 lane 间交换；
- per-lane result writeback。

其端口不是“任意 bank直接连接任意 lane FU”，而是每 lane 提供一个 64-bit operand/result stream，并在 lane 内与 AddrGen、ALU/MFPU reduction 共享/仲裁。

### 15.2 MASKU

MASKU 集中处理：

- mask bit 分发；
- mask logical/comparison result；
- `vfirst`、`vcpop` 等 scalar result；
- `vrgather/vrgatherei16/vcompress` 的控制与特殊 operand 请求；
- 向各 lane 提供最终 byte mask。

MASKU 与 SLDU、VLSU 一样通过有限 queue 和 ready/valid 连接所有 lanes。

### 15.3 Ara 风格 reduction

禁止使用理想、无冲突的独立 tree reduction 代替 Ara。推荐状态机：

```text
Phase 1: each lane performs intra-lane sequential/SIMD reduction
Phase 2: lane partial results enter SLDU
Phase 3: inter-lane exchange/combination
Phase 4: lane 0 performs final SIMD reduction
Phase 5: result writes VRF or returns scalar path
```

需要显式建模 lane workload 不均衡、SLDU bus 仲裁、ALU/MFPU reduction busy，以及下一条 reduction/VALU 指令的互斥条件。

## 16. Completion、Squash 与 Drain

### 16.1 多层 completion

```text
word result ready
  != VRF write committed
  != target VFU instruction done
  != all targets done
  != Minor commit
```

每个 target VFU 维护 instruction queue 的 issue/commit counters。算术指令的 VFU done 必须等最后一个有效 result final grant；load 必须等 result queue 排空和最后一次 VRF final grant；store 按所选 precise-store policy 完成。

### 16.2 Squash

所有状态同时携带 `contextId/araId`。squash 时：

- Dispatcher 和 pending broadcast 移除 younger instructions；
- lane sequencer、requester metadata 和 operand queues 按 Ara ID flush；
- FU 内不可取消结果在输出时丢弃；
- VRF 不接受 killed instruction writeback；
- outstanding load response继续吸收；
- 未释放 store buffer 取消；
- running bitmap、hazard matrix、VFU counters 和 context 引用一致释放；
- 内部 reshuffle/segment micro-op 与父指令一起取消。

### 16.3 Drain

`isDrained()` 必须检查：

```text
Dispatcher/SegmentSequencer empty
Global Sequencer running bitmap empty
pending broadcast empty
all lane requesters IDLE
all operand queues empty
all FU pipelines/result streams empty
all VRF read pipelines empty
VLSU burst/request/result/outstanding/store buffers empty
SLDU/MASKU idle
CompletionQ empty
no scheduled RVV event
```

## 17. 推荐类层级

```text
Minor::Execute
└── RvvEngine
    ├── RvvIssueAdapter
    ├── AraDispatcher
    │   ├── VectorCsrSnapshot
    │   ├── EewTagTable
    │   ├── ReshuffleSequencer
    │   └── SegmentSequencer
    ├── AraGlobalSequencer
    │   ├── AraIdAllocator
    │   ├── RunningBitmap
    │   ├── GlobalHazardMatrix
    │   └── TargetVfuCounters
    ├── VectorLane[NLanes]
    │   ├── AraLaneSequencer
    │   ├── OperandRequester[NrOperandQueues]
    │   ├── ResultProgressTable
    │   ├── AraVectorRegFile
    │   │   ├── LaneOrganizationMapper
    │   │   ├── VrfBankArbiter[8]
    │   │   └── SinglePortVrfBank[8]
    │   ├── OperandQueue[NrOperandQueues]
    │   ├── VectorAlu
    │   ├── VectorMfpu
    │   └── ResultStream
    ├── AraVlsu
    │   ├── AddrGen
    │   ├── VectorLoadUnit
    │   ├── VectorStoreUnit
    │   ├── LoadResultQueue
    │   ├── MemoryShuffle
    │   └── TimingMemoryAdapter
    ├── AraSldu
    ├── AraMasku
    └── CompletionTracker
```

推荐文件：

```text
src/cpu/minor/rvv/
├── ara_types.hh
├── rvv_engine.hh/.cc
├── ara_dispatcher.hh/.cc
├── ara_global_sequencer.hh/.cc
├── ara_lane.hh/.cc
├── ara_lane_sequencer.hh/.cc
├── ara_operand_requester.hh/.cc
├── ara_operand_queue.hh/.cc
├── ara_vrf.hh/.cc
├── ara_vrf_mapper.hh/.cc
├── ara_vfu.hh/.cc
├── ara_vlsu.hh/.cc
├── ara_sldu.hh/.cc
├── ara_masku.hh/.cc
└── rvv_completion.hh/.cc
```

## 18. 推荐参数

### 18.1 Ara baseline profile

| 参数 | Ara 对齐默认值 | 说明 |
|---|---:|---|
| `LaneWidthBits` | 64 | 固定 lane word |
| `NLanes` | 2/4/8/16 | power-of-two profile |
| `VrfBanksPerLane` | 8 | baseline 固定 |
| `VrfBankPorts` | 1 shared R/W | 单端口 |
| `VrfReadLatency` | 1 | request 后一周期返回 |
| `VrfBankPolicy` | AraTwoLevelPriority | high/low 两级仲裁 |
| `NrAraInsnIds` | 与目标 RTL profile 一致 | running slots |
| `GlobalIssueWidth` | 1 instruction request/cycle | 不生成多个全局 beat |
| `ChainingMode` | VrfWriteProgress | 默认模式 |
| `DirectFuBypass` | false | 非 Ara baseline |
| `EewTags` | true | reshuffle 所需 |
| `MemoryShuffle` | true | natural/lane organization |
| `SlduAllLanePorts` | true | 每 lane 64-bit stream |
| `ReductionMode` | IntraLaneThenSldu | 禁止理想 tree |
| `VlsuAccessModes` | unit/stride/index/segment | P1 必需 |
| `FaultOnlyFirst` | true | P1 必需 |

FU latency、queue depth、AXI/timing memory width和 VLEN 必须从选定 Ara配置记录在单独 profile 中。不得把“推荐值”伪装成 Ara RTL 的固定事实。

### 18.2 研究扩展 profile

以下参数允许探索，但默认关闭，并在统计输出中标记：

- bank 数量或端口类型变化；
- age-based 取代 Ara 两级 priority；
- direct FU bypass；
- ideal/tree reduction；
- cache-line ideal coalescer；
- 多 instruction/cycle global issue；
- vector renaming 或乱序；
- 独立 RVV clock domain。

## 19. 分阶段实现

### 19.1 P0：接口和资源正确

- Minor accept/completion/squash/drain 接通；
- Ara ID allocator、running bitmap、instruction-level broadcast；
- global RAW/WAR/WAW matrix，但依赖先整指令阻塞；
- lane sequencer 与 operand requester FSM；
- 64-bit lane word、8-bank single-port VRF、1-cycle read；
- Ara 两级 bank arbitration；
- operand queues、ALU/MFPU、result stream、真实 final grant；
- unit-stride load/store timing requests；
- 不做 chaining，所有依赖等待 producer 完成；
- slide/reshuffle 可先序列化，但必须走显式 SLDU latency；
- 完整 current/next、trace、assertion。

### 19.2 P1：Ara 周期行为对齐目标

P1 完成后才能宣称“Ara-like cycle behavior”。必须实现：

- global sequencer 为每个 operand 生成 producer hazard bitmap；
- 每 operand requester 独立维护 address、remaining、EEW 和 hazard metadata；
- 每 lane、每 Ara ID 的真实 VRF write progress；
- consumer 按 producer write rate逐 64-bit word chaining；
- widening/narrowing 的 write/read 比率与 WAW toggle；
- 普通依赖不使用直接 FU bypass；
- 多条无依赖/可 chaining 指令在 ALU、MFPU、VLSU 等不同目标中 overlap；
- Lane Organization mapper；
- EEW tag、load/store shuffle/deshuffle；
- Dispatcher 自动插入 reshuffle micro-op；
- masked、tail-undisturbed 和 old-vd 的真实 VRF traffic；
- SLDU slide、reshuffle 和 inter-lane reduction；
- MASKU mask/scalar/gather/compress 协调；
- VLSU unit、strided、indexed、segment、fault-only-first；
- Ara burst descriptor到 gem5 Packet 的适配；
- load result queue、partial result drain、exception 和 `vstart`；
- Ara 式 target done、result final grant 和 hazard release；
- 对所有关键结构提供与 Ara RTL trace 可比较的周期事件。

### 19.3 P2：研究扩展

- 非 Ara baseline 的 direct bypass；
- alternative bank mappings、更多 banks 或多端口 SRAM；
- ideal/coalesced memory profile；
- multiple global issue；
- parameterized cross-lane topology；
- register cache、load-to-use forwarding；
- speculative store buffer优化；
- vector renaming、乱序或多线程；
- 不同 CPU/RVV 时钟域。

P2 结果必须与 P1 Ara baseline 分开报告。

## 20. 周期示例：Ara 式 producer-consumer Chaining

假设同一 lane 中：

```asm
vadd.vv v3, v1, v2
vmul.vv v4, v3, v5
```

VRF read=1，ALU latency=1，MUL latency=2，无 bank conflict。示意时序：

```text
Cycle 0  global sequencer broadcasts VADD, allocates ID 0
Cycle 1  lane sequencer creates ALU operand requests and operation
Cycle 2  VADD src word0 reads fire
Cycle 3  VADD word0 operands enter queues; src word1 reads fire
Cycle 4  VADD word0 enters ALU; global sequencer broadcasts VMUL ID 1
Cycle 5  VADD word0 result waits/gets VRF write grant
Cycle 6  VADD word0 write visible, progress[0]=1
         VMUL requester may now read v3 word0
Cycle 7  VMUL v3 word0 response enters operand queue
         VADD word1 write visible, progress[0]=2
         VMUL requester may read v3 word1
Cycle 8  VMUL word0 enters MUL
Cycle 10 VMUL word0 result valid and competes for VRF write
```

如果 VADD word0 在 cycle 5 因目标 bank 被高优先级请求占用，则 `progress[0]` 不增加，VMUL 对 v3 word0 的请求也必须继续 stall。不能在 ALU result ready 时提前解除依赖。

## 21. 统计与 Trace

### 21.1 关键事件

```text
DISPATCH_ACCEPT
ARA_ID_ALLOC
PE_BROADCAST_START / TARGET_ACCEPT / COMPLETE
OPERAND_CMD_ACCEPT
REQUESTER_STALL_HAZARD / STALL_OPQ / STALL_BANK
VRF_READ_FIRE / READ_RESP
VFU_WORD_ISSUE / RESULT_VALID
VRF_WRITE_GRANT / RESULT_WRITTEN / FINAL_GRANT
VLSU_BURST / PACKET / LOAD_RESULT / EXCEPTION
SLDU_PHASE / MASKU_PHASE / RESHUFFLE
TARGET_DONE / RVV_COMPLETE / MINOR_COMMIT
```

统一字段：

```text
cycle, contextId, araId, lane, operandSlot, wordIndex,
module, event, bank, targetVfu, stallReason
```

### 21.2 必要统计

- Ara ID occupancy 和 global issue stall；
- hazard matrix density；
- 每 requester hazard/opqueue/bank stall cycles；
- producer write progress 与 chaining distance；
- 每 bank read/write/conflict/priority starvation；
- operand queue occupancy；
- ALU/MFPU utilization、latency、II、output blocking；
- reshuffle 数量和周期；
- shuffle/deshuffle bytes 和 stall；
- SLDU/MASKU busy、lane traffic、reduction phases；
- VLSU burst、Packet、retry、result queue 和 exception；
- Minor dispatch gap 对 RVV utilization 的影响。

## 22. 验证与 Ara 对拍

### 22.1 三层验证

1. 功能：与 RVV ISA reference 比较 architectural result；
2. 局部周期：对 mapper、requester、bank arbiter、OpQ、VFU、VLSU 写单元测试；
3. 系统周期：对相同 instruction stream 比较 Ara Verilator trace 与 MinorRVV event trace。

### 22.2 对拍归一化事件

两边信号名不同，按以下事件归一化：

| 归一化事件 | Ara RTL | MinorRVV |
|---|---|---|
| instruction accepted | dispatcher/sequencer handshake | `PE_BROADCAST_COMPLETE` |
| operand word request | `vrf_req` read grant | `VRF_READ_FIRE` |
| operand available | delayed VRF valid | `VRF_READ_RESP` |
| VFU word accepted | VFU ready/valid | `VFU_WORD_ISSUE` |
| result produced | VFU result request | `RESULT_VALID` |
| result committed | VRF write grant/final grant | `RESULT_WRITTEN` |
| target instruction done | per-VFU done bit | `TARGET_DONE` |
| full instruction done | sequencer response | `RVV_COMPLETE` |

允许 gem5 memory latency与 AXI testbench 不同，但在固定 memory responder profile 下，AddrGen、burst、result queue 和 VRF write事件应可比较。

### 22.3 必要断言

- 一个 Ara ID 不能同时属于两条 active instruction；
- target 不能重复 sample 同一 pending broadcast；
- single-port bank 每周期最多一次 request fire；
- requester 只有在 bank grant 后才推进 address/remaining；
- consumer 读取 dependent word 时，producer write progress 必须覆盖该 word；
- FU result ready 不得直接清除普通 hazard；
- final grant 必须发生在最后一个有效 VRF write 提交之后；
- EEW 不匹配且需保留数据时必须存在 reshuffle；
- killed instruction 不得产生 VRF/scalar/memory architectural side effect；
- drain 为 true 时所有 ID、queue、pipeline、outstanding 和 event 均为空。

## 23. 推荐实现顺序

1. 定义 `AraInsnId`、running bitmap、hazard matrix 和 instruction-level `PeRequest`；
2. 接通 Minor accept/completion/squash/drain；
3. 实现 global sequencer 广播和 per-target accepted bitmap；
4. 实现 lane sequencer、9 类 operand request commands 和 VFU operation；
5. 实现固定 8-bank、64-bit、单端口、1-cycle VRF；
6. 实现 requester metadata FSM 与 Ara 两级 bank arbitration；
7. 实现 operand queues、ALU/MFPU 和 final-grant writeback；
8. 实现 result progress table和 word-level chaining；
9. 加入 widening/narrowing progress ratio；
10. 实现 Lane Organization、EEW tag 和 reshuffle；
11. 实现 SLDU、MASKU 和 Ara reduction 状态机；
12. 实现 AddrGen、VLDU/VSTU、shuffle、indexed/segment/FOF；
13. 完成异常、squash、drain 和随机 backpressure；
14. 建立 Ara Verilator 与 MinorRVV 归一化 trace 对拍；
15. 冻结 P1 baseline 后再启用 P2 研究扩展。

## 24. 设计结论

Ara 对齐版 MinorRVV 的核心不是把一条向量指令预先拆成若干全局 beat，而是复现以下流式控制关系：

```text
instruction-level global scheduling
        ↓
per-operand lane-local request streaming
        ↓
banked single-port VRF contention
        ↓
operand queue decoupling
        ↓
64-bit SIMD VFU execution
        ↓
real VRF write progress
        ↓
dependent operand chaining
```

因此最终边界为：

```text
MinorCPU：architectural instruction、scalar dependency、squash、commit

MinorRVV：Ara ID、hazard matrix、lane requester、VRF banks、OpQ、VFU、
          VLSU、SLDU、MASKU、result progress 和 vector completion
```

当 P1 列出的 write-progress chaining、Lane Organization、EEW/reshuffle、Ara bank arbitration、VLSU 和 SLDU/MASKU 行为全部实现并通过归一化 trace 对拍后，该模型才应称为“Ara-like cycle-accurate model”。在此之前应称为“Ara-inspired timing model”。

## 25. Ara 参考实现映射

| 本文模块 | Ara 参考 |
|---|---|
| Dispatcher | [ara_dispatcher.sv](https://github.com/pulp-platform/ara/blob/main/hardware/src/ara_dispatcher.sv) |
| Global Sequencer | [ara_sequencer 文档](https://pulp-platform.github.io/ara/modules/ara_sequencer.html) |
| Lane | [lane 文档](https://pulp-platform.github.io/ara/modules/lane/lane.html) |
| Lane Sequencer | [lane_sequencer.sv](https://github.com/pulp-platform/ara/blob/main/hardware/src/lane/lane_sequencer.sv) |
| Operand Requester / chaining | [operand_requester.sv](https://github.com/pulp-platform/ara/blob/main/hardware/src/lane/operand_requester.sv) |
| VRF banks | [vector_regfile.sv](https://github.com/pulp-platform/ara/blob/main/hardware/src/lane/vector_regfile.sv) |
| VRF layout/reshuffle | [Ara VRF 文档](https://pulp-platform.github.io/ara/modules/lane/vrf.html) |
| VLSU | [VLSU 文档](https://pulp-platform.github.io/ara/modules/vlsu/vlsu.html) |
| SLDU | [SLDU 文档](https://github.com/pulp-platform/ara/blob/main/docs/source/modules/sldu/sldu.md) |
| RVV 1.0 Ara 论文 | [A New Ara for Vector Computing](https://arxiv.org/abs/2210.08882) |
