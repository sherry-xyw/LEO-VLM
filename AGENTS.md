# AGENTS.md

## Project Scope

This workspace supports a Chinese research survey on on-orbit multimodal/VLM foundation models for LEO computing constellations. The intended paper positioning is an agenda-defining survey for venues such as IEEE Communications Surveys & Tutorials, focused on deployment, collaborative inference, continual adaptation, and reliability under space constraints.

## Workspace Map

- `综述架构论证/综述架构.md`: primary survey outline, positioning, taxonomy, chapter plan, and writing requirements.
- `参考文献/参考文献.md`: meeting-friendly reference table organized by clear sections. It summarizes key papers with link, year, journal/source, first author and affiliation, keywords, and short summary.
- `参考文献/参考文献_参考.md`: detailed evidence ledger, including verified PDFs, candidate references, competitive survey analysis, and citation status.
- `参考文献/References/`: local PDF library. Treat a paper as usable only after title, authors, venue/source, and relevance have been checked against the PDF or an authoritative source.
- `chatGPT.md`: prior brainstorming and planning notes. Use as background, not as authoritative citation evidence.
- `会议讨论/`: meeting notes and current task assignments.
- `.claude/settings.local.json`: local Claude permission history; do not treat it as project documentation.

## Writing Conventions

- Default writing language is Chinese for project notes and drafts.
- Preserve the paper's core positioning: "on-orbit multimodal/VLM foundation models as reliability-critical distributed intelligence over LEO computing constellations."
- Keep the distinction between VLM/multimodal remote-sensing workloads, generic LLM workloads, SAGIN/NTN communication surveys, and Earth-observation model-only surveys.
- Maintain the agenda-defining framing: the field is early, so maturity gaps and missing on-orbit evidence should be stated honestly rather than hidden.
- Prefer structured tables and taxonomies when comparing literature, tasks, constraints, models, hardware, or evaluation metrics.

## Citation And Evidence Rules

- Do not add or rely on a citation solely from memory or a chat transcript.
- Before treating a reference as verified, confirm at least title, authors, source/venue, and arXiv/DOI or publication identifier.
- For newly used papers, add the PDF to `参考文献/References/` when available and update `参考文献/参考文献_参考.md` with download/status notes.
- Keep status labels consistent with `参考文献/参考文献_参考.md`: downloaded/local PDF, open access not downloaded, paywalled, unverified, or author/source conflict.
- Be especially careful with future-dated or preprint references. If the current date matters, record the exact verification date.
- If a claim comes from `chatGPT.md`, re-check it against primary sources before moving it into the main outline or bibliography.

## Research Priorities

- Protect the three differentiation claims in `综述架构论证/综述架构.md`: VLM-first rather than LLM-first, deployment-reliability-continuity rather than algorithm cataloging, and explicit hardware constraint tiers.
- When expanding the survey, map additions back to the TCM framework: Task, Constraint, Model.
- Separate single-satellite deployment, constellation-scale collaborative inference, continual/on-orbit adaptation, and model/hardware/network reliability.
- For evaluation discussions, include accuracy, latency, energy, memory, downlink saved, AoI, calibration, fault tolerance, and mission-level risk where relevant.

## Current Technical Directions

Meeting notes on 2026-06-01 define four concrete technical directions. Use the Ch.5-Ch.8 names in `综述架构论证/综述架构.md` as the preferred names:

- 单星部署：轻量化 VLM 与硬件感知. Covers quantization, pruning, distillation, sparse inference, KV cache compression, streaming inference, early exit, and hardware-aware deployment under satellite compute and power limits.
- 星座协同：模型切分、激活路由与拓扑感知调度. Covers model splitting across satellites, intermediate activation compression, inter-satellite scheduling, large-small model collaboration, and interruption-tolerant inference.
- 持续适应：在轨微调. Covers RAG updates, prompt/adapter updates, LoRA, federated LoRA, drift detection, asynchronous aggregation, and multi-satellite memory merging.
- 可靠性：辐射、故障与可信推理. Covers weight, activation, KV cache, RAG index, communication, and node faults, plus selective fault tolerance, confidence estimation, anomaly detection, radiation-aware training, and reliable collaborative inference.

Current assignment:

- 汪雪怡: 单星部署：轻量化 VLM 与硬件感知; 持续适应：在轨微调.
- 黎增懋: 星座协同：模型切分、激活路由与拓扑感知调度; 可靠性：辐射、故障与可信推理.

## Reference Document Style

- If creating a simplified reference document, keep the structure clear and vocabulary plain.
- Avoid dense compound labels and long argumentative paragraphs.
- Use these survey sections: `综述（星座智能与在轨VLM）`, `综述（空间网络智能）`, `综述（遥感基础模型与遥感 VLM）`, `综述（对地观测模型与中文综述）`, and `综述（硬件与可靠性）`.
- Hardware and reliability papers may also appear under `可靠性：辐射、故障与可信推理` when they support Ch.8 directly.
- After the survey sections, use the four Ch.5-Ch.8 technical directions as the main method sections.
- Each literature section should use a table with these columns: link, year, journal/source, first author and affiliation, keywords, and short summary.
- Include papers already present in `参考文献/References/` before adding external candidates.
- Keep `参考文献/参考文献_参考.md` as the detailed evidence ledger; keep `参考文献/参考文献.md` as the clean meeting and collaboration document.

## Operational Notes

- This folder is currently not a git repository. Do not assume git history or branch workflows are available.
- Avoid modifying or renaming PDFs unless explicitly requested.
- Keep edits scoped. Do not reorganize the literature library, change citation statuses, or rewrite the outline wholesale unless the task asks for it.
- Use `rg`/`rg --files` for workspace searches.
