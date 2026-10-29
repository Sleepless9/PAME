# DMA descriptor-ring and explicit buffer ownership

> 更新：本文记录早期 MMIO runtime 方案，不能作为已完成硬件 ring 的依据。
> 最新实现、准确参数语义与限制见 [DMA_HARDWARE_RING.md](DMA_HARDWARE_RING.md)。
> 旧版 `dma_regs[10]` 是活动 ID，不能用于 completion fence；现已修复为新模型
> 的连续完成 sequence，旧模型使用 busy 等待。下面“本次改造已验证”的措辞不代表全模型已验证。

本次改造把算子侧 DMA 从“每次提交后立即 blocking wait”扩展为可提交多个 descriptor、按依赖 fence 的 runtime。baseline 仍使用 `qwen_dma_copy()`，因此原有阻塞口径不变；启用 overlap 的算子使用 `qwen_dma_submit()` 和 `qwen_dma_fence()`。

## 数据路径

```text
CPU runtime → SPMVirtDma MMIO descriptor queue
           → DDR/system port ↔ coherence cache ↔ SystemXBar ↔ DDR
           → SPM port ↔ shared BankedScratchpad
           → AME/RVV local SPM port
```

AME 和 RVV 不直接访问 DDR。CPU 只负责提交 descriptor；搬运、burst 切分、SPM bank 仲裁和 outstanding credit 由 `SPMVirtDma` 完成。

## 当前接口

```c
uint64_t qwen_dma_submit(uintptr_t ddr, uintptr_t spm,
                         unsigned bytes, unsigned spm_to_ddr);
void qwen_dma_fence(uint64_t descriptor_id);
unsigned qwen_dma_buffer_index(unsigned slot);
```

`submit` 只写 descriptor 并启动，不等待完成。`fence(id)` 等待指定 descriptor 以及它之前的 descriptor 完成，并检查 DMA error。descriptor id 由 runtime 单调递增，设备队列按提交顺序退休。

## 双缓冲生命周期

```text
slot 0: DMA filling  ── fence ── AME consumes ── output DMA
slot 1:                         DMA filling ── fence ── AME consumes
```

算子必须遵守以下 ownership：

1. CPU/DDR buffer 在对应 descriptor fence 前保持只读，不能复用或覆盖。
2. DMA 写入 SPM slot 时，AME/RVV 不得读取该 slot。
3. `fence(slot_n)` 完成后，slot_n 才转移给 AME/RVV。
4. AME/RVV 计算完成后，slot_n 才能重新交给 DMA 或作为输出 DMA 源。
5. 相邻 slot 使用不同 SPM bank stripe，避免把双缓冲退化成同 bank 排队。

## outstanding 和 burst

SoC 当前参数为 64 B burst、descriptor queue 8、每 bank outstanding 4、system outstanding 16。runtime 最多保持两个输入 slot 和一个输出 descriptor 在 DMA 队列中；设备内部继续按 burst 和 SPM interleave 切分请求。增大 outstanding 只有在 DDR 延迟能被计算覆盖、且 `bankWaitTicks`/`bankQueueFull` 没有明显上升时才有收益。

## 后续硬件化方向

当前仍是 CPU 写 MMIO 的 descriptor queue。更完整的硬件 command queue 可以把 AME command、RVV command 和 DMA descriptor 放入同一个依赖图，由 DMA completion event 唤醒 AME，而不是由 CPU polling。该方向需要新增 event/doorbell 或 descriptor ring 的模拟器模型；本次 runtime 改造只实现了可验证的 descriptor 提交、fence 和双 buffer ownership。
