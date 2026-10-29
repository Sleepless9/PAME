# 基于 gem5 MinorCPU 的 MinorAME P1 周期精确建模方案

> 目标级别：MinorRVV P1 / Ara-style resource-accurate timing model  
> ISA 语义：Ztt Matrix Extension v0.6  
> 计算结构：多 tile Systolic Array  
> 数据流：Weight-Stationary 与 Partial-Sum-Stationary  
> 集成对象：gem5 MinorCPU Execute、InstQueue、in-flight、commit、squash、drain

## 1. 建模目标与边界

本文是一份可直接用于 C++ 开发、SimObject 参数定义、代码评审、trace 对拍和性能实验的 MinorAME 实现说明。目标不是给矩阵指令设置一个固定 `opLat`，而是让周期数从真实资源行为自然产生：

- Minor InstQueue 与 AME Dispatcher 的 ready/valid；
- 有限 AME instruction slots、running bitmap 和 hazard matrix；
- M register group、Acc read-modify-write 和 fragment-level dependency；
- MRF/Acc RF 单端口 bank 竞争；
- operand requester、MOFU、有限队列和 A/B double buffer；
- physical tile、PE MAC pipeline、tile-edge FIFO 和 SA wavefront；
- weight preload、Acc seed、K-step、drain 和 architectural commit；
- M-LSU timing memory、retry、outstanding transaction 和 result writeback；
- reduction/scan 的局部计算与跨 tile 合并；
- completion、squash、drain 和 current/next state。

本文复用 MinorRVV_Ara P1 的控制思想，但不声称复制 Ara 的具体 lane RTL：

```text
Ara instruction ID       -> AmeInsnId
Ara global sequencer     -> Matrix Global Sequencer
Ara lane sequencer       -> SA Sequencer + Tile Sequencer
VRF word requester       -> MRF/Acc fragment requester
VRF resultWritten        -> fragmentCommitted / accCommitted
lane-local VRF banks     -> banked MRF / banked Acc RF
operand queue            -> A/B/Acc queues and tile buffers
VFU pipeline             -> injection / PE MAC / edge / drain pipeline
VLSU / SLDU / MASKU      -> M-LSU / MOFU / MRSU / optional MPTU
```

P1 达标后可以称为“Ara-style AME timing model”或“resource-accurate MinorAME”。由于没有一个与本文结构相同的公开 AME RTL 作为逐信号 golden reference，不应直接声称“Ara cycle-equivalent”。

## 2. Ztt architectural state 的实现解释

### 2.1 Logical square

```text
N_AME     = amenlen
AME_NELEM = N_AME * N_AME
```

`N_AME x N_AME` 是 ISA logical square。它不是 physical tile 大小，也不是整个 SA 大小。

### 2.2 M register

M register 是固定宽度 raw-bit container：

```text
MREG_BITS = AME_UNIT_DATATYPE_SIZE * AME_NELEM
```

当前 `Md` 决定 raw bits 如何形成 square：

```text
Md < unit width   一个 M register 含多个 logical squares
Md = unit width   一个 M register 正好含一个 square
Md > unit width   一个 square 跨多个连续 M registers
```

MRF 因而不能定义为 `MRegister[N_AME][N_AME]`。底层保存固定 bytes，MOFU 根据 `Md`、square index 和 register group 形成逻辑矩阵。

### 2.3 Acc register

一个 architectural Acc register 始终只表示一个 `N_AME x N_AME` square：

```text
ACC_ENTRY_BITS = AME_NELEM * MAX_ACC_ELEMENT_BITS
```

窄 `Ad` 不允许在一个 Acc 中打包多个 square；宽 `Ad` 也不跨多个 Acc registers。物理 Acc entry 按实现支持的最大 `Ad` 设计。

### 2.4 Acc 与 partial sum

```text
PE psumReg             microarchitectural
Tile edge/psum FIFO    microarchitectural
SA resident state      microarchitectural backing state
Architectural Acc RF   architectural state
```

architectural Acc RF 位于整个 AME/SA 的 ingress/egress 边界。每个 PE 保存执行中的 partial sum；tile 内部边界只放 elastic register/FIFO。只有最终阵列边界或独立 sub-array 边界执行 Acc drain/commit。

### 2.5 矩阵乘

`mmulacc.2d` 类指令建模为：

```text
Acc[acc] = Acc[acc] + A * B
```

若 operand formation 产生 `N_SQ` 个 source squares：

```text
A aggregate shape = N_AME x (N_SQ * N_AME)
B aggregate shape = (N_SQ * N_AME) x N_AME
K_LOGICAL          = N_SQ * N_AME
Acc shape          = N_AME x N_AME
```

## 3. 物理参数与命名

禁止使用同一个 `N` 同时表示 ISA square、physical tile 和 SA：

```text
N_AME               architectural square side
PE_ROWS, PE_COLS     one physical tile shape
TILE_ROWS, TILE_COLS tile grid shape
SA_ROWS              PE_ROWS * TILE_ROWS
SA_COLS              PE_COLS * TILE_COLS
K_BLOCK              one prefetched K block
```

示例：

```text
N_AME = 16
tile  = 8 x 8 PEs
grid  = 2 x 2 tiles
SA    = 16 x 16 PEs
```

若 `SA_ROWS/SA_COLS < N_AME`，一条指令产生多个 output blocks；若大于 `N_AME`，SA 可分为多个独立 partitions，同时执行不同 square 或不同 instruction。

## 4. 建模粒度

| 粒度 | 产生者 | 消费者 | 完成条件 |
|---|---|---|---|
| Architectural instruction | Minor/Dispatcher | Global Sequencer | 所有 required targets done |
| Instruction request | Global Sequencer | SA、M-LSU、MRSU、MPTU | required targets accepted |
| Output block | SA Sequencer | tile partitions | 所有 K blocks 和 drain 完成 |
| K block | SA Sequencer | tile sequencers | required fragments 可用且 tile 接受 |
| K step | Tile Sequencer | ingress/PE | A/B/psum input 真实 fire |
| Tile fragment | tile/edge requester | adjacent tile | edge FIFO 接受 |
| MRF fragment | requester | MRF bank | bank grant + read/write visibility |
| PE operation | tile ingress | MAC pipeline | psum state 更新 |
| Acc fragment | Acc requester | Acc bank/SA distributor | read response或write commit |
| Memory descriptor | M-LSU | gem5 timing port | response/side effect完成 |

Global Sequencer 不预展开全部 K-step。它按整条 instruction 广播；SA Sequencer 和 Tile Sequencer 在 backpressure 下流式推进。

## 5. 总体结构

```text
                             MinorCPU
                    Fetch / Decode / Execute
                                |
                         InstQueue issue
                                | ready/valid
                                v
                  +-----------------------------+
                  | AME Adapter / Dispatcher    |
                  | snapshot / legality / route |
                  +--------------+--------------+
                                 | AmeInstToken
                                 v
                  +-----------------------------+
                  | Matrix Global Sequencer     |
                  | ID / running / hazards      |
                  | target accept/done          |
                  +--------------+--------------+
                                 | AmePeRequest broadcast
          +----------------------+-----------------------+
          |                      |                       |
          v                      v                       v
   +--------------+       +--------------+       +--------------+
   | SA Sequencer |       | M-LSU       |       | MRSU / MPTU  |
   | MOFU route   |       | timing mem  |       | reduce/perm  |
   +------+-------+       +------+-------+       +------+-------+
          |                      |                      |
          +----------------------+----------------------+
                                 |
                           Banked MRF
                                 |
                   requesters / arbiter / response
                                 |
                      +----------+----------+
                      |                     |
                    A DBuf                B DBuf
                      |                     |
                      +----------+----------+
                                 |
                  +-----------------------------+
                  | Tile Grid                   |
                  | TileSeq / edge FIFOs / PEs  |
                  | weightReg / psumReg / MAC   |
                  +--------------+--------------+
                                 |
                         Acc drain/combiner
                                 |
                          Banked Acc RF
                                 |
                         target completion
                                 |
                       Minor in-flight/commit
```

## 6. 与 MinorCPU 的集成

### 6.1 责任边界

| MinorCPU | MinorAME |
|---|---|
| fetch、decode、branch | AME decode 补充、target route |
| scalar dependency | M/Acc dependency 与 chaining |
| `MinorDynInst`、PC、seqNum | `AmeInsnId`、block/K-step tags |
| architectural in-flight/commit | MRF/SA/Acc/M-LSU 内部执行 |
| squash、interrupt、drain request | token flush、response 吸收 |

一条 AME instruction 在 Minor 中始终只对应一个 `MinorDynInst`。内部 K-step、tile fragment 和 memory packet 不进入 Minor in-flight queue。

### 6.2 Issue handshake

```cpp
bool fire = isAmeInst(inst) && ame.canAccept(inst);
if (fire) {
    ame.acceptNext(inst, scalarSnapshot, ameConfigSnapshot);
    minorMarkIssued(inst);
}
```

`canAccept()` 联合检查 Dispatcher/Context/ID 空间、register group legality、scalar ready、配置序列化、precise-store、drain/squash 和必要的全局资源条件。只有 `fire` 后 Minor 才能移除指令。

### 6.3 Snapshot

dispatch 保存：

```text
N_AME, Md, Ad, pack/wide mode
transposeA, transposeB, dataflow
M/Acc operands
memory base/stride/shape
rounding/saturation mode
scalar operands
```

后端不得读取之后被配置指令修改的 AME state。

## 7. Context、ID 与 Token

```cpp
using AmeContextId = uint64_t;
using AmeInsnId = uint8_t;

struct AmeInstContext {
    AmeContextId contextId;
    AmeInsnId ameId;
    Minor::MinorDynInstPtr minorInst;
    ThreadID tid;
    uint64_t streamSeqNum;
    uint64_t predictionSeqNum;

    AmeOpcode opcode;
    AmeTargetMask targetMask;
    AmeConfigSnapshot cfg;
    MOperandDesc ms1, ms2, md;
    AccIndex acc;

    MRegHazardSet sourceHazards;
    MRegHazardSet destHazards;
    AccHazardSet accHazards;
    AmeTargetMask acceptedTargets;
    AmeTargetMask completedTargets;
    Fault fault;
    bool killed;
};
```

`contextId` 用于 gem5 生命周期和 squash；有限 `ameId` 用于 running bitmap、hazard matrix、requester metadata 和 progress table。

### 7.1 Instruction-level request

```cpp
struct AmePeRequest {
    AmeContextId contextId;
    AmeInsnId ameId;
    AmeOpcode opcode;
    AmeTargetMask targetMask;
    MOperandDesc ms1, ms2, md;
    AccIndex acc;
    AmeConfigSnapshot cfg;
    DynamicBitset hazardMs1, hazardMs2, hazardMd, hazardAcc;
};
```

这是整条 AME instruction，不是一个 beat。

### 7.2 Local tags

```cpp
struct SaBlockTag {
    AmeContextId contextId;
    AmeInsnId ameId;
    uint16_t outputBlockRow, outputBlockCol, kBlock;
};

struct TileTag {
    SaBlockTag block;
    TileId tile;
    uint16_t kStep;
    MatrixPhase phase;
};

struct MFragmentTag {
    AmeContextId contextId;
    AmeInsnId ameId;
    OperandSlot slot;
    uint16_t squareIndex, logicalRow, fragmentIndex;
};
```

## 8. Dispatcher 与 Global Sequencer

Dispatcher 负责完整 decode、config snapshot、M register group/square index 解析、`N_SQ/K_LOGICAL` 计算、Acc read-modify-write 标记、target route 和 transpose flags。它不访问 MRF。

Global Sequencer 状态：

```cpp
struct AmeGlobalSequencerState {
    DynamicBitset running;
    std::array<AmeContextId, NrAmeInsnIds> owner;
    std::array<AmeTargetMask, NrAmeInsnIds> accepted;
    std::array<AmeTargetMask, NrAmeInsnIds> done;
    MRegHazardMatrix mHazards;
    AccHazardMatrix accHazards;
};
```

职责：

- 分配有限 AME ID；
- 保存 running bitmap；
- 建立 RAW/WAR/WAW producer bitmap；
- 向 required targets 广播 `AmePeRequest`；
- 用 per-target accepted bitmap 防止重复接收；
- 收集 target completion；
- 释放 ID/hazards并向 Minor 返回 completion。

Global Sequencer 不得直接生成 bank/row、K-step 或 PE operation，也不得用预计完成周期替代内部执行。

## 9. Dependency 与 progress

### 9.1 M register regions

P1 支持：

```text
ConservativeGroup  任意重叠 M register 按整寄存器 hazard
SquareRegion       跟踪 (mreg, byteBegin, byteEnd)
```

默认 `SquareRegion`；partial write、未知覆盖或 datatype reinterpretation 时退化为整寄存器。

### 9.2 MRF write progress

```cpp
struct MWriteProgress {
    uint32_t fragmentsCommitted;
    uint32_t totalFragments;
    bool finalGrantSeen;
};
```

只有 destination bank write 在周期边界可见后才递增。Consumer requester按所需 byte coverage 检查 producer progress。transpose/column stream 不能简单比较 fragment ordinal，必须用 `ProducerCoverageMap` 判断所需 bytes 是否已提交。

### 9.3 Acc ownership

```cpp
enum class AccLocation {
    AccRF, LoadingToArray, ResidentInArray, Draining, CommitPending
};

struct AccProgress {
    bool busy;
    AmeInsnId owner;
    AccLocation location;
    uint32_t elementsLoaded;
    uint32_t elementsUpdated;
    uint32_t elementsCommitted;
    uint16_t kBlocksCompleted;
    bool finalGrantSeen;
};
```

规则：

- `mmulacc` 独占对应 Acc 的 read-modify-write ownership；
- PE resident 时，Acc RF 中旧副本不能被普通 consumer读取；
- `mzero.2d.acc`、`mmov.m.a` 和后续 Acc writer 观察同一 hazard；
- 默认在最终 Acc commit 后释放；
- 可选 `AccResidentHandoff` 只允许相邻、同 Acc、无中间观察者、mapping兼容的指令直接接管，默认关闭。

### 9.4 分布式 tile progress

不使用理想化万能 Tile Scoreboard。真实 ready 来自：

```text
SA Sequencer       phase/output/K block
Tile Sequencer     inject/consume/drain counters
MRF Requester      address/remaining/hazard/progress
Operand Buffer     occupancy/valid bitmap
Edge FIFO          accepted fragments
PE state           weightValid/psumValid/MAC pipeline
Acc Progress       resident/drain/commit
```

`TileProgressMirror` 只用于 debug/assertion。

## 10. SA Sequencer 与 Tile Sequencer

SA Sequencer 计算：

```text
outputBlocksM = ceil(N_AME / activeSaRows)
outputBlocksN = ceil(N_AME / activeSaCols)
kBlocks       = ceil(K_LOGICAL / K_BLOCK)
```

状态机：

```text
ALLOCATE
 -> FORM_OPERANDS
 -> PREFETCH_A_B
 -> LOAD_WEIGHTS or LOAD_ACC
 -> WAVEFRONT
 -> NEXT_K_BLOCK
 -> DRAIN
 -> ACC_COMMIT
 -> DONE
```

Tile command：

```cpp
struct TileCommand {
    SaBlockTag tag;
    TileId tile;
    MatrixDataflow dataflow;
    MatrixPhase phase;
    uint16_t validRows, validCols, validK;
    bool firstKBlock, lastKBlock;
};
```

每个 tile 的 sequencer维护 local A/B/weight/Acc buffer command、row/column injection、edge ready、PE mode、K-step accepted 和 drain progress。只有最后一个有效 result 离开 tile 后才发 `TileDone`。

## 11. MRF 微结构

### 11.1 Baseline

```text
MrfBanks             8 or 16, configurable
MrfBankWordBits      64 or 128
MrfBankPorts         1 shared read/write
MrfReadLatency       1 cycle
MrfWriteVisibility   next cycle
```

选择多 bank single-port SRAM，而不是 monolithic dual-port MRF。A/B double buffers 解耦 MRF 与 SA，bank conflict通过仲裁自然产生。

### 11.2 Fixed raw-bit layout

先算 register group absolute byte，再算：

```text
wordAddress = absoluteByte / MRF_WORD_BYTES
bank        = wordAddress % MrfBanks
row         = wordAddress / MrfBanks
byteOffset  = absoluteByte % MRF_WORD_BYTES
```

`Md` 只影响 element boundary、square selection和byte enable，不改变 MRF raw bits 排列。

### 11.3 Request types

```cpp
struct LogicalMrfReq {
    MFragmentTag tag;
    MAccess access;
    MRegIndex baseReg;
    uint64_t operandByteOffset;
    uint16_t byteCount;
    MDataType md;
    ByteMask byteEnable;
    MatrixQueueId targetQueue;
};

struct MrfBankReq {
    MFragmentTag tag;
    MAccess access;
    BankId bank;
    RowId row;
    BankWord writeData;
    ByteMask byteEnable;
    MatrixQueueId targetQueue;
};

struct MrfReadResp {
    MFragmentTag tag;
    MatrixQueueId targetQueue;
    BankWord data;
    ByteMask validBytes;
};
```

MOFU、M-LSU、MRSU、MPTU 和 result writeback 都经过同一个 arbiter。

### 11.4 Arbitration

```text
High:
  active-wavefront operand read
  architectural result writeback
  active reduction/scan traffic

Low:
  next-block prefetch
  M-LSU load writeback/store read
  optional MPTU
```

组内 round-robin。未获 grant 的 requester保持 address、remaining、valid 和 tag。公平提升只能作为独立研究 profile。

## 12. MOFU 与 operand formation

MOFU 负责：

- 从一个 M register 中选择 packed square；
- 从多个 M registers拼接 wide square；
- 为 `N_SQ > 1` 形成扩展 K stream；
- 根据 transpose flags 改变 logical访问顺序；
- 生成边界 valid masks；
- 将 MRF words 整形成 SA row/column fragments；
- 必要时使用有限 transpose/line buffer。

```text
normal A:     A[i][k]
transpose A:  A[k][i]
normal B:     B[k][j]
transpose B:  B[j][k]
```

transpose matrix-multiply variant默认不改写 MRF，只改变 requester 与 feeder。按列读取造成的 bank conflict必须真实建模。

```cpp
struct OperandFormerState {
    AmeInsnId ameId;
    OperandSlot slot;
    uint16_t squareIndex, logicalRow, logicalCol, kIndex;
    uint32_t bytesRequested, bytesReceived;
    DynamicBitset sourceHazards;
    bool transpose;
};
```

## 13. Requester、Queue 与 Double Buffer

独立 requester：

```text
SA_A / SA_B / SA_ACC_SEED
MRSU_SRC0 / MRSU_OLD_DST
MPTU_SRC
MLSU_STORE_SRC / MLSU_SHAPE
```

```cpp
struct MatrixRequesterState {
    AmeInsnId ameId;
    OperandSlot slot;
    uint64_t nextLogicalByte;
    uint32_t remainingBytes, nextFragment;
    DynamicBitset hazards;
    MatrixQueueId targetQueue;
    RequesterPhase phase;
};
```

A/B double buffer：

```text
A0 active, A1 prefetch
B0 active, B1 prefetch
```

每个 buffer 有 owner、block tag、fragment valid bitmap、produced/consumed counters和有限容量。只有 active 已消费、next 已到齐、tile ready 且 tag 匹配时才能 swap。

Acc RF read 进入有限 `AccSeedQueue`，再由 distributor 分发至 PE/tile。队列满时反压 Acc requester。

## 14. Tile、PE 和 inter-tile network

### 14.1 PE state

```cpp
struct PeState {
    MatrixElement weightReg;
    AccElement psumReg;
    bool weightValid;
    bool psumValid;
    Pipeline<MacToken> macPipe;
};
```

`weightReg` 按最大支持 weight element宽度实现，`psumReg` 至少支持 architectural maximum Acc element width；额外 guard bits允许作为内部精度，但在 Acc commit 时执行规定的 round/truncate/saturate。

### 14.2 Links

```text
Horizontal: A or activation west -> east
Vertical:   B or partial sum north -> south
```

每个 PE hop 和每个 tile boundary 都必须参数化：

```text
PeHopLatency
TileBoundaryLatency
EdgeFifoDepth
MacLatency
MacInitiationInterval
```

tile-edge FIFO 使用 ready/valid。下游不 ready 时，payload和valid保持；具体采用 per-stage stall、elastic pipeline 还是只冻结末级，由参数 `SaBackpressureModel` 定义。

### 14.3 PE operation token

```cpp
struct MacToken {
    TileTag tag;
    PeCoord pe;
    MatrixElement a;
    MatrixElement bOrWeight;
    AccElement psumIn;
    bool valid;
};
```

PE result ready不等于 Acc ready，也不等于 instruction done。

## 15. Weight-Stationary

### 15.1 Mapping

```text
B weights preload and remain in PE.weightReg
A flows west -> east
partial sum flows north -> south
```

### 15.2 Phases

```text
WS_PRELOAD_WEIGHT
  MOFU B requester -> B buffer -> PE weightReg

WS_LOAD_ACC_SEED
  Acc RF/zero -> north psum ingress

WS_STREAM_A
  each K-step inject A rows from west

WS_FORWARD_PSUM
  psums traverse PE and internal tile boundaries

WS_DRAIN
  only final south/partition boundary enters Acc drain
```

内部 tile 边界不写 architectural Acc，只通过 `PsumEdgeFifo` 转发。若 tile partition独立执行不同 output square，每个 partition可以连接一个 distributed Acc bank，但仍由全局 Acc index/tag管理。

### 15.3 Weight lifetime

weight有效期至少覆盖当前 `K_BLOCK`。进入下一 K block前，必须确认：

- 当前 weight不再被 in-flight MAC 使用；
- 新 weight fragments 已到齐；
- preload目标 PE全 ready；
- weight-valid bitmap在周期边界统一更新。

不得在最后一个 A 注入时立即覆盖 weight；必须等待对应 wavefront 消费完成。

## 16. Partial-Sum-Stationary

### 16.1 Mapping

```text
A flows west -> east
B flows north -> south
psum remains in PE.psumReg across K steps/blocks
```

### 16.2 Phases

```text
PS_LOAD_ACC
  old Acc/zero distributed to PE psumReg

PS_STREAM_AB
  synchronized A/B K-step injection

PS_ACCUMULATE
  each PE updates local psumReg

PS_NEXT_K_BLOCK
  retain psum, swap A/B buffers

PS_DRAIN
  after final K block, drain PE psums to Acc combiner
```

PSS 模式下 `AccProgress.location = ResidentInArray`。只要某个 PE 的 psum尚未安全 drain，Acc owner不能释放。

### 16.3 A/B alignment

A 和 B 必须携带相同 block/K-step tag。某一侧迟到时不能消费另一侧。可用一个有限 `PeOperandJoin`：

```cpp
struct PeOperandBundle {
    TileTag tag;
    MatrixElement a, b;
    bool aReady, bReady;
};
```

只有 `aReady && bReady && macPipe.ready()` 才 issue。

## 17. Acc RF、drain 与 commit

### 17.1 Physical organization

推荐 Acc RF 也使用 banked storage：

```text
AccBanks             4/8/16 configurable
AccBankWordBits      64/128/256
AccBankPorts         1 shared read/write baseline
AccReadLatency       1 cycle
AccWriteVisibility   next cycle
```

由于每条 matrix multiply只在开始读取 Acc seed、结束写入 Acc result，多个 single-port banks加 ingress/egress queue足够。`1R1W` 作为吞吐研究 profile，不作为 baseline。

### 17.2 Drain path

```text
PE/tile result
 -> Tile Drain Queue
 -> SA Drain Network
 -> Acc Write Combiner
 -> Acc Bank Mapper
 -> Acc Bank Arbiter
 -> Acc RF
```

`AccWriteCombiner` 将 PE/tile fragments 整形成 Acc bank words，并生成 byte enables。若多个 tile写同一 Acc element，必须有明确的 K-partition reduction；默认禁止多个 partitions无仲裁写同一 Acc element。

### 17.3 Commit event

```text
PE psum ready
 != drain queue accepted
 != Acc bank write grant
 != Acc fragment committed
 != Acc instruction done
 != Minor commit
```

只有最后一个有效 Acc bank write在周期边界可见后产生 `ACC_FINAL_GRANT`，SA target才可 done。

## 18. M-LSU

### 18.1 Structure

```text
AmePeRequest
 -> Matrix Address/Shape Generator
 -> Burst Descriptor Queue
 -> gem5 Packetizer
 -> sendTimingReq / retry
 -> outstanding table
 -> recvTimingResp
 -> load reorder/format queue
 -> MRF write requester
```

### 18.2 Descriptor

```cpp
struct MatrixBurstDesc {
    AmeInsnId ameId;
    uint64_t descId;
    Addr base;
    uint32_t row, col;
    uint32_t rows, cols;
    int64_t rowStride, colStride;
    uint16_t elementBytes;
    bool isLoad, isStore;
    MRegIndex mBase;
    uint64_t mByteOffset;
};
```

一个 descriptor可映射为多个 gem5 packets。completion按 descriptor bytes、outstanding、load result queue和最终 MRF grants判断，不能把最后一个 Packet 返回时间直接当作指令完成。

### 18.3 Row/column layout

Memory row/column-major和stride由 M-LSU address generation决定。load时可直接按照目标 raw-bit layout写 MRF，不需要经过 MPTU。矩阵乘的 `A^T/B^T` 则由 MOFU read/routing决定，不改写 MRF。

### 18.4 Precise store

P1 baseline：

- unresolved branch后禁止 vector/matrix store产生不可撤销 memory side effect，或使用 commit-gated store buffer；
- killed load response仍接收并移除 outstanding，但不得写 MRF；
- store完成定义必须明确为“进入不可撤销队列”或“memory response完成”；baseline采用后者；
- fault保存最早 faulting element/row/byte位置。

## 19. MRSU 与 optional MPTU

### 19.1 MRSU

MRSU 处理 row/column reduction和prefix/scan，因为它们包含真实计算，不能仅靠访存顺序实现。

```text
Phase 0  request source fragments
Phase 1  tile/local reduction or local scan
Phase 2  cross-tile exchange/merge
Phase 3  ordered prefix propagation if required
Phase 4  format result
Phase 5  MRF/Acc/scalar writeback
```

Reduction推荐两级：

```text
within tile -> tile partial
tile partials -> finite merge network -> result
```

Prefix必须保持顺序，使用 scan state和有限 pipeline，不能替换成无延迟理想树。

### 19.2 MPTU

MPTU仅在支持显式 register-to-register permutation/transpose/reshape时存在：

```text
MRF read -> permutation network/transpose buffer -> MRF write
```

内存布局变换属于 M-LSU；matrix-multiply transpose flag属于 MOFU；不要为了普通 `A^T/B^T` 强制增加一次 MRF read-modify-write。

## 20. Completion

### 20.1 Layers

```text
MRF/Acc fragment committed
 -> tile/K-block done
 -> output block done
 -> target unit done
 -> AME instruction complete
 -> Minor architectural commit
```

### 20.2 Completion tracker

```cpp
struct TargetProgress {
    AmeTargetMask required;
    AmeTargetMask accepted;
    AmeTargetMask done;
};
```

算术指令：最后 Acc/MRF final grant后 SA/MRSU target done。Load：所有 response格式化并最终写入 MRF。Store：按 precise-store policy。Scalar result：结果进入 Minor scalar completion path。

## 21. Squash 与 drain

### 21.1 Squash

每个 token携带 `contextId/ameId/streamSeqNum`。squash：

1. Context标记 killed；
2. Dispatcher/Global Sequencer取消未广播 request；
3. SA/Tile sequencer停止新 K-step；
4. requester、operand buffers、edge/drain queues按 ID flush；
5. killed result不得写 MRF/Acc/scalar state；
6. 已发出的 load仍吸收 response；
7. 已进入不可撤销路径的 store按 precise-store policy处理；
8. resident Acc若尚未 architectural commit可丢弃，Acc RF旧值必须仍可恢复；
9. 释放 AME ID前确认所有不可取消 transaction已归档。

为了精确恢复，默认禁止 speculative instruction覆盖 Acc RF。计算结果保持于 PE/drain queue，直到 instruction达到允许写 architectural candidate state的条件；或者为 Acc RF提供版本/tag并在 commit选择版本。

### 21.2 Drain conditions

```text
Dispatcher empty
Global running bitmap empty
all target queues empty
SA/Tile sequencers idle
A/B/Acc seed buffers empty
all PE MAC pipelines empty
all edge/drain FIFOs empty
MRF/Acc read pipelines empty
M-LSU outstanding == 0
completion queue empty
no resident dirty Acc state
```

`MinorCPU::isDrained()` 必须加入 `ame.isDrained()`。

## 22. Ready/valid 与 current/next

### 22.1 Channel

```cpp
template<class T>
class RvChannel {
  public:
    bool valid() const;
    bool ready() const;
    bool fire() const { return valid() && ready(); }
    const T &payload() const;
    void pushNext(const T &);
    void popNext();
    void advance();
};
```

规则：

- `valid && !ready` 时 payload稳定；
- 只有 `fire` 才发生一次 transfer；
- 本周期写 next，下周期才能消费；
- FIFO、spill register、buffer均有限容量；
- broadcast用 per-target accepted bitmap；
- fanout到多个 tile时，采用 all-required-ready，或保存 per-tile accepted bitmap，不能重复发送。

### 22.2 Current/next

每周期：

1. 只读 current state/channel；
2. 计算 ready、valid、grant、fire；
3. 只写 next state/channel；
4. 周期边界统一 commit；
5. 新状态下周期可见。

```cpp
void MinorAme::evaluate()
{
    completion.evaluate();
    accWriteback.evaluate();
    mlsu.evaluate();
    mrsu.evaluate();
    tileArray.evaluate();
    operandSubsystem.evaluate();
    saSequencer.evaluate();
    globalSequencer.evaluate();
    dispatcher.evaluate();
}

void MinorAme::advance()
{
    channels.advance();
    modules.commitNextState();
}
```

交换 `evaluate()` 调用顺序若会改变结果，说明 current/next隔离不完整。

### 22.3 Latency semantics

```text
MRF read request fire at Cn       -> response valid Cn+1
MRF write grant at Cn            -> visible/progress Cn+1
PE input fire at Cn               -> result after MacLatency
tile-edge fire at Cn              -> next tile visible after boundary latency
Acc write grant at Cn             -> committed Cn+1
```

`latency` 与 `initiation interval`独立。MAC可 `latency=3, II=1`。

## 23. P1 周期示例：PSS

假设：

```text
N_AME = 4
SA = one 4x4 tile
K_LOGICAL = 4
MRF read = 1
Acc read = 1
MAC latency = 1, II = 1
PE hop = 1
无 bank conflict
```

示意事件：

```text
C0  Minor issue fire
C1  Dispatcher allocates context/ID
C2  Global broadcast accepted by SA
C3  MOFU/MRF A,B requests begin; Acc read request begins
C4  first MRF/Acc responses enter buffers
C5  Acc distributor starts PE psum initialization
C6  A/B k0 ingress fire where valid
C7  k1 ingress; k0 moves/executes
C8  k2 ingress; earlier wavefront advances
C9  k3 ingress
... wavefront tail drains through PE/tile latency ...
Cx  final PE psum enters drain queue
Cx+1 Acc bank writes compete/grant
Cy  final Acc write visible -> ACC_FINAL_GRANT
Cy+1 SA target done -> AME completion
```

如果 MRF bank冲突，A/B fragment不到齐，K-step不得注入；如果 Acc writeback bank被占用，`ACC_FINAL_GRANT`和 instruction complete必须后移。禁止在最后一个 K-step注入时直接宣布完成。

## 24. P1 周期示例：WS 多 tile

假设 psum由北向南跨两个 tile：

```text
Tile0 south edge -> FIFO -> Tile1 north edge
```

必须观察：

```text
Tile0 PE result valid
 != edge FIFO accepted
 != Tile1 consumed
 != Tile1 south result
 != Acc drain accepted
 != Acc committed
```

Tile1或 edge FIFO backpressure时，Tile0不得丢失或重复推进 psum。内部边界不写 architectural Acc。

## 25. 推荐类层级

```text
MinorAme
├── AmeIssueAdapter
├── AmeDispatcher
├── AmeContextTable
├── AmeGlobalSequencer
│   ├── AmeIdAllocator
│   ├── MRegHazardMatrix
│   ├── AccHazardMatrix
│   └── TargetCompletionTracker
├── MatrixOperandSubsystem
│   ├── Mofu
│   ├── MrfAddressMapper
│   ├── MrfBankArbiter
│   ├── MatrixRegisterFile
│   │   └── MrfBank[N]
│   ├── MatrixOperandRequester[*]
│   ├── ADoubleBuffer
│   ├── BDoubleBuffer
│   └── AccSeedQueue
├── SystolicBackend
│   ├── SaSequencer
│   ├── SaPartitionScheduler
│   ├── TileGrid
│   │   └── MatrixTile[R][C]
│   │       ├── TileSequencer
│   │       ├── IngressQueues
│   │       ├── PeArray[PR][PC]
│   │       ├── HorizontalEdgeFifo
│   │       ├── VerticalEdgeFifo
│   │       └── TileDrainQueue
│   ├── SaDrainNetwork
│   └── AccWriteCombiner
├── AccumulatorSubsystem
│   ├── AccRegisterFile
│   ├── AccBankMapper
│   ├── AccBankArbiter
│   ├── AccProgressTable
│   └── AccCommitQueue
├── MatrixLsu
│   ├── MatrixAddrGen
│   ├── BurstDescQueue
│   ├── Gem5Packetizer
│   ├── OutstandingTable
│   └── LoadResultQueue
├── MatrixReductionScanUnit
├── OptionalMatrixPermutationUnit
├── AmeCompletionQueue
└── AmeTraceStats
```

推荐文件：

```text
src/computing_cluster/AME/
  ame_types.hh
  ame_channels.hh
  ame_engine.hh/.cc
  ame_dispatcher.hh/.cc
  ame_sequencer.hh/.cc
  ame_scoreboard.hh/.cc
  mrf.hh/.cc
  mofu.hh/.cc
  operand_requester.hh/.cc
  systolic_array.hh/.cc
  matrix_tile.hh/.cc
  pe.hh/.cc
  acc_rf.hh/.cc
  matrix_lsu.hh/.cc
  mrsu.hh/.cc
  mptu.hh/.cc
  ame_trace.hh/.cc
```

## 26. 推荐参数

### 26.1 P1 baseline profile

| 参数 | 默认建议 | 说明 |
|---|---:|---|
| `NrAmeInsnIds` | 4 | 有限 active instruction slots |
| `N_AME` | 来自 ISA 配置 | architectural square side |
| `PeRows/PeCols` | 4/8 可配置 | physical tile |
| `TileRows/TileCols` | 1/2/4 | tile grid |
| `MrfBanks` | 8 或 16 | single-port banks |
| `MrfBankWordBits` | 64/128 | fragment width |
| `MrfReadLatency` | 1 | request到response |
| `MrfWriteVisibility` | 1 | grant到progress visible |
| `AccBanks` | 4/8 | single-port baseline |
| `AccReadLatency` | 1 | seed read |
| `OperandBufferCount` | 2 per A/B | double buffer |
| `OperandQueueDepth` | 2-4 fragments | finite |
| `EdgeFifoDepth` | 1-2 | tile boundary |
| `PeHopLatency` | 1 | per hop |
| `TileBoundaryLatency` | 1 | registered boundary |
| `MacLatency` | datatype profile | 与 II 独立 |
| `MacII` | 1 baseline | fully pipelined MAC |
| `MrfArbPolicy` | two-level + RR | Ara-style |
| `AccResidentHandoff` | false | 可选优化 |
| `DirectPeBypass` | false | 非必要通路 |
| `Dataflows` | WS/PSS | 两种均实现 |
| `PreciseStorePolicy` | response-complete | 保守 baseline |

具体 PE/MAC latency、bank width、queue depth必须记录在独立 profile；不能把推荐值伪装成某个 RTL 的固定事实。

### 26.2 Research profile

- 1R1W MRF/Acc banks；
- wider prefetch/commit bandwidth；
- `AccResidentHandoff`；
- direct PE-to-next-op bypass；
- fair/age-based bank arbitration；
- multi-instruction SA partitioning；
- alternative dataflows；
- multicast/compression network。

这些结果必须与 P1 baseline分开报告。

## 27. Trace、统计与断言

### 27.1 Normalized events

```text
AME_DISPATCH_ACCEPT
AME_ID_ALLOC
AME_GLOBAL_BROADCAST / TARGET_ACCEPT
MRF_REQ / MRF_GRANT / MRF_RESP
MRF_WRITE_GRANT / M_FRAGMENT_COMMITTED / M_FINAL_GRANT
MOFU_FRAGMENT_FORMED
BUFFER_ENQ / BUFFER_SWAP / BUFFER_STALL
TILE_CMD_ACCEPT
WEIGHT_PRELOAD
ACC_SEED_LOAD
KSTEP_INJECT
PE_MAC_ISSUE / PE_MAC_RESULT
EDGE_FIFO_FIRE / EDGE_STALL
TILE_DRAIN / ACC_WRITE_GRANT / ACC_FRAGMENT_COMMITTED
ACC_FINAL_GRANT
TARGET_DONE / AME_COMPLETE / MINOR_COMMIT
MLSU_REQ / RETRY / RESP
SQUASH / DRAIN_BEGIN / DRAIN_END
```

### 27.2 Stats

- AME ID occupancy、global issue stall原因；
- M/Acc RAW/WAR/WAW等待周期；
- requester hazard/bank/queue stall；
- MRF/Acc bank conflict按source分类；
- A/B buffer利用率和swap等待；
- weight preload、Acc seed、wavefront、drain各阶段周期；
- PE/MAC利用率；
- edge FIFO occupancy/backpressure；
- WS与PSS每指令cycles和energy proxy；
- M-LSU outstanding、retry、bytes/cycle；
- MRSU local/cross-tile阶段周期；
- squash丢弃的micro-ops和吸收的responses。

### 27.3 Required assertions

- 一个 `AmeInsnId` 不同时属于两个 active contexts；
- valid&&!ready时payload稳定；
- requester未获grant不得推进address/count；
- killed instruction不得写 MRF/Acc/scalar state；
- M fragment ready只能由真实 committed coverage解除；
- `ACC_FINAL_GRANT`发生在最后一个 Acc write visible之后；
- tile不能消费tag不匹配的A/B/psum；
- PSS下未初始化 `psumReg`不得MAC；
- WS下未初始化 `weightReg`不得MAC；
- internal tile boundary不得产生 architectural Acc commit；
- drain前所有 required K steps已完成；
- completion前required targets全done；
- drain返回 true时无resident dirty Acc、无outstanding memory。

## 28. 验证方案

### 28.1 Functional

- 对每种 `Md/Ad` 验证 M register group到square formation；
- packed multiple-square和wide cross-register；
- normal、A transpose、B transpose、both transpose；
- WS/PSS与软件矩阵乘golden result；
- Acc zero、old Acc accumulation、M<->Acc move；
- row/column reduction和prefix；
- boundary blocks、invalid PE masks；
- load/store stride和layout。

### 28.2 Resource timing

- 构造同-bank A/B conflict；
- writeback阻塞prefetch；
- edge FIFO满导致上游停止；
- Acc commit阻塞instruction completion；
- memory retry和out-of-order response；
- producer fragment晚写导致consumer requester停在精确fragment；
- transpose column stream产生可解释的MRF conflict；
- 不同queue depth改变吞吐但不改结果。

### 28.3 Wavefront oracle

理想无阻塞时，可以用近似式做 sanity check：

```text
T_wavefront approximately K_effective + activeRows + activeCols - 2
```

它只能作为验证下界/预期，不得作为执行模型。真实 latency必须由 preload、hop、tile boundary、MAC latency、drain和bank arbitration事件产生。

### 28.4 RTL/trace 对拍

若后续选定具体 SA RTL，建立归一化事件映射：

| RTL event | MinorAME event |
|---|---|
| command handshake | `TILE_CMD_ACCEPT` |
| SRAM read grant/data | `MRF_GRANT/MRF_RESP` |
| operand buffer valid | `BUFFER_ENQ` |
| array injection | `KSTEP_INJECT` |
| MAC pipeline valid | `PE_MAC_ISSUE/RESULT` |
| tile boundary valid | `EDGE_FIFO_FIRE` |
| accumulator write | `ACC_WRITE_GRANT` |
| command done | `TARGET_DONE` |

## 29. P0/P1/P2

### 29.1 P0：资源正确

- Minor issue/completion/squash/drain；
- finite AME IDs和instruction-level hazards；
- fixed raw-bit MRF、多 bank single-port、1-cycle read；
- one instruction at a time in SA；
- MOFU normal square formation；
- WS/PSS基本phase和PE result正确；
- Acc seed/drain/final grant；
- unit/strided M-LSU timing request；
- current/next、ready/valid和trace。

### 29.2 P1：本文目标

必须实现：

- 多 active AME instructions和target broadcast；
- M register region hazard和producer bitmap；
- fragment/byte-coverage write progress chaining；
- Acc ownership、resident、drain和commit progress；
- packed/wide M operand formation；
- transpose-aware MOFU和有限transpose buffer；
- A/B double buffering和prefetch/compute overlap；
- per-tile sequencer、K-step stream、PE MAC latency/II；
- tile-edge FIFO与真实backpressure；
- WS与PSS完整数据流；
- MRF/Acc RF真实bank arbitration和write final grant；
- M-LSU descriptors、packetization、retry、reorder和precise store；
- MRSU row/column reduce及prefix状态机；
- squash/drain覆盖所有内部状态；
- 关键事件trace、stats和assertions。

### 29.3 P2：研究扩展

- multi-instruction SA partitioning；
- resident Acc跨指令handoff；
- speculative/versioned Acc；
- advanced direct bypass；
- adaptive dataflow选择；
- multiported/hierarchical MRF；
- NoC式tile network；
- gather/scatter和复杂permutation；
- OoO block scheduling。

## 30. 推荐实现顺序

1. 定义 `AmeInsnId`、Context Table、running bitmap和target mask；
2. 接入 Minor issue/completion/squash/drain；
3. 实现 M/Acc decode、register-group和square mapping单元测试；
4. 实现 fixed raw-bit MRF mapper；
5. 实现 single-port banks、1-cycle read和arbiter；
6. 实现 MOFU normal/packed/wide formation；
7. 实现 A/B double buffers和requester FSM；
8. 实现单 tile PE array、PSS、Acc seed/drain；
9. 实现 WS weight preload和vertical psum；
10. 扩展到多 tile、edge FIFO和registered boundaries；
11. 实现 MRF fragment progress和chaining；
12. 实现 Acc Progress Table和final grant；
13. 实现 transpose-aware MOFU；
14. 实现 M-LSU timing memory和precise store；
15. 实现 MRSU reduction/scan；
16. 加入完整trace/stats/assertions；
17. 建立功能golden、资源冲突和周期oracle测试；
18. 冻结P1 baseline后再开启P2 profile。

## 31. P1 完成判据

只有满足以下条件才称为 MinorAME P1：

1. 改变 MRF/Acc bank数会通过真实冲突改变周期；
2. 改变 queue/FIFO深度会通过backpressure改变吞吐；
3. 改变 tile/PE/MAC参数会通过wavefront事件改变周期；
4. transpose的额外成本来自实际访问顺序、bank conflict和buffer，不是固定penalty；
5. consumer只在producer所需byte coverage真实commit后推进；
6. Acc在PE resident、drain、bank commit和instruction complete之间有清晰状态；
7. WS内部tile边界只转发psum，最终边界才commit Acc；
8. PSS跨全部K blocks保持PE psum，最后统一drain；
9. M-LSU延迟来自gem5 timing memory、retry、response和MRF writeback；
10. 随机squash/drain不泄漏ID、buffer、resident Acc或outstanding transaction；
11. 每个额外周期都能由trace中的hazard、bank、queue、edge、MAC、memory或commit事件解释。

## 32. 与 MinorRVV_Ara 的最终对应

| MinorRVV_Ara P1 | MinorAME P1 |
|---|---|
| Dispatcher snapshots VTYPE/VL | Dispatcher snapshots Md/Ad/N/dataflow |
| Ara ID/running bitmap | AME ID/running bitmap |
| global hazard matrix | M region + Acc hazard matrices |
| PeRequest broadcast | AmePeRequest broadcast |
| lane sequencer | SA/Tile sequencers |
| VRF word requester | MRF/Acc fragment requester |
| EEW/lane organization | Md/square/register-group formation |
| 8-bank lane-local VRF | configurable banked MRF/Acc RF |
| operand queues | MOFU queues + A/B/Acc buffers |
| ALU/MFPU word stream | K-step/PE MAC wavefront |
| resultWritten progress | M/Acc fragment committed progress |
| SLDU/MASKU reduction | MOFU/MRSU/optional MPTU |
| VLSU AXI descriptor | M-LSU matrix burst descriptor |
| final VRF grant | M/Acc final grant |

核心原则不变：

> 整条 instruction由全局控制管理；局部 requester根据真实依赖和资源逐片推进；计算结果必须经过真实存储写回/驻留提交才能解除依赖；所有延迟都从有限结构、仲裁和backpressure中产生。

## 33. 参考

- Ztt-v0.6：AME architectural state、M/Acc、matrix multiply、move、reduction和prefix语义。
- [Ara top-level documentation](https://pulp-platform.github.io/ara/)
- [Ara lane documentation](https://github.com/pulp-platform/ara/blob/main/docs/source/modules/lane/lane.md)
- [Ara VRF documentation](https://pulp-platform.github.io/ara/modules/lane/vrf.html)
- [Ara lane sequencer RTL](https://github.com/pulp-platform/ara/blob/main/hardware/src/lane/lane_sequencer.sv)
- [A New Ara for Vector Computing](https://arxiv.org/abs/2210.08882)
