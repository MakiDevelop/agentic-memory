# LinkedIn Post — memcite v1.0.0

## 版本 A：六位一體協作故事（推薦）

---

剛剛把我的 open source library **memcite v1.0.0** 發佈到 PyPI。

這次改版有點特別——**五個新子系統、一千多行新 code、108 個新測試，是我跟六位 AI agent 一起寫完的**。

先講問題：

你的 AI agent 記得「這個專案用 Jest 寫測試」。兩週後同事換成 Vitest，agent 不知道，還在寫 Jest 測試，把 CI 搞壞了。

這不是 hallucination——記憶**本來是對的**。它是 **stale memory**，比 hallucination 更麻煩，因為 agent 對它的錯誤有十足把握。

memcite 的解法：**每一條記憶都必須標註出處，使用前自動驗證——像給 agent context 寫 unit tests**。

```python
mem.add("Uses ruff for linting",
        evidence=FileRef("pyproject.toml", lines=(15, 20)))

# 同事改了 pyproject.toml 之後：
mem.validate()  # → ⚠ 1 memory stale (evidence changed)
```

---

**v1.0.0 的五個新子系統**：

🧩 **Plugin Architecture** — 可擴展的 evidence types、storage、search backends（entry_points）
🕸️ **Memory Graph** — 記憶之間可以有 contradicts / supports / supersedes / depends_on 關係
🧠 **Semantic Search** — 選配 sentence-transformers / ONNX embedding，和 TF-IDF 共存
🌐 **Cross-repo Federation** — 跨 repo 查詢，本地優先，零外部服務
♻️ **Lifecycle Automation** — 自動過期 / 降級 / 壓縮 + git pre-commit hook

**從 188 → 296 tests、13 → 22 modules、schema v5 → v7，100% 向下相容。**

---

**這次最爽的不是 feature，是協作方式**：

- **Claude** 負責架構設計 + 實作
- **Gemini** 做競品深度分析（mem0 / Zep / Letta / LangMem），幫我把 memcite 定位在「evidence-first knowledge governance」這個真空地帶
- **Gemma4 (local)** 跑 codebase 影響分析
- **Perplexity Max** 查 sentence-transformers / entry_points 的最新最佳實踐
- **我** 做方向決策 + 四層北極星把關 + 按 publish 鍵

一天內 5 個 phase 全部寫完、commit、push、PyPI 發佈。

這是我第一次真正感受到「X 位一體協作」不是 buzzword，而是實際可運作的工程流程：**不是讓一個 AI 包辦所有事情，而是讓每個 agent 做它最擅長的事**。

---

📦 `pip install --upgrade memcite`
🔗 https://github.com/MakiDevelop/agentic-memory
📄 https://pypi.org/project/memcite/1.0.0/

把 agent 當同事教，就得有人幫忙 code review agent 的記憶——這就是 memcite 想做的事。

#AIAgents #OpenSource #Python #LLM #AgentInfrastructure #DevTools

---

## 版本 B：純技術導向（較短）

---

memcite v1.0.0 上線了——AI agent 的**記憶治理層**，強制引用 + 自動過期偵測。

AI agent 的 stale memory 比 hallucination 更危險：它「本來是對的」，所以 agent 對錯誤答案有十足把握。memcite 強迫每條記憶標註出處，使用前自動驗證源頭是否變動——像給 context 寫 unit tests。

**v1.0.0 新增**：

- Plugin Architecture（可擴展 evidence types / storage backends）
- Memory Graph（contradicts / supports / supersedes / depends_on）
- Semantic Search（sentence-transformers / ONNX，和 TF-IDF 共存）
- Cross-repo Federation（跨 repo 查詢，local-first）
- Lifecycle Automation（auto expire / downgrade / compact + git pre-commit hook）

188 → 296 tests，13 → 22 modules，100% 向下相容。

`pip install --upgrade memcite`

https://github.com/MakiDevelop/agentic-memory

#AIAgents #OpenSource #Python #LLM
