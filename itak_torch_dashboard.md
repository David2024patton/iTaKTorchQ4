# iTaK Torch Dashboard Architecture

Inspired by Unsloth Studio (AGPL-3.0), this document maps out the architecture for building a native, no-code web GUI wrapper around the `iTaK Torch` native Go fine-tuning CLI.

## Core Philosophy

Unsloth Studio demonstrated massive demand for a visual interface over complex fine-tuning scripts. We are adopting their UI/UX paradigms while completely ripping out the buggy Python/Triton underneath, replacing the backend exclusively with the lightning-fast, zero-dependency `torch.exe`.

## Feature Integrations

### 1. Live Loss & Reward Telemetry Graphs

**Unsloth Inspiration:** Real-time charting of GRPO rewards and Supervised Fine-Tuning loss.
**iTaK Implementation:** 
Instead of complex websocket streaming, the UI will poll the native `--status-file` (e.g., `training_status.json`) generated automatically by `train_orchestrator.go`. The Next.js dashboard will use Recharts or Chart.js to map `avg_reward` and `loss` dynamically every 500ms.

### 2. Drag-and-Drop Dataset Conversion

**Unsloth Inspiration:** No-code data ingestion for PDFs, CSVs, and Word docs.
**iTaK Implementation:** 
A Next.js frontend zone where users drop files. A local Node.js server (or intermediate Go server) runs parsing and translates unstructured text into the strict ChatML array required by `torch train`: 
`{"messages": [{"role": "user", "content": "..."}]}`

### 3. Visual Hyperparameter Sliders

**Unsloth Inspiration:** Slider bars for learning rates, LoRA ranks, and epochs.
**iTaK Implementation:**
The UI provides simple HTML range sliders. When the user clicks "Start Training", the frontend executes the backend process seamlessly:

```bash
./torch.exe train --model <selected.gguf> --data training.jsonl --mode <sft|grpo> --lr <slider_val> --lora-rank <slider_val> --epochs <slider_val>
```

### 4. Side-by-Side Model Inference Testing

**Unsloth Inspiration:** Comparing the base model versus the newly fine-tuned model.
**iTaK Implementation:**
Because `torch train` automatically calls `trainer.MergeAndExport()` and outputs a clean standalone `.gguf`, the Dashboard simply launches two parallel `iTaK Torch` inference daemon slots on different ports. The UI sends a test prompt to both and visualizes the Orchestrator JSON reasoning improvements perfectly.

## Proposed Tech Stack

- **Frontend Dashboard:** Next.js (React), Tailwind CSS (or simple GoHighLevel style aesthetics)
- **Backend Bridge:** Express.js daemon (or a lightweight Go Fiber layer) to execute `os/exec` on `torch.exe`
- **Charting Engine:** Recharts (React)
- **Core ML Engine:** Pure `iTaK Torch` (Native Go executing GGUFs)

## Future Integration: Exhaustive Training Paradigms Roadmap

To transform the `iTaK Torch` Dashboard into the ultimate competitor to Unsloth Studio, here is what it would take to build native Go support for both legacy and bleeding-edge Machine Learning methodologies.

### Bleeding-Edge Architectures (2025/2026)

#### 1. DoRA (Weight-Decomposed LoRA)

- **What it is:** The strict upgrade to LoRA. It splits adapter weights into Magnitude and Direction for vastly superior training stability without catastrophic forgetting.
- **What it takes (Go Engine):** **High Effort.** We must map the inner Go tensor multiplication logic in the base Engine to natively calculate magnitude/direction vectors independently during the forward/backward pass.
- **What it takes (Dashboard):** A simple toggle switch next to the LoRA rank slider: `[x] Enable DoRA`.

#### 2. Medusa/Lookahead Heads (Speculative Decoding)

- **What it is:** Training 3-5 extra "heads" on top of the model to predict multiple future tokens simultaneously, yielding 3x inference speeds directly out of the box.
- **What it takes (Go Engine):** **Extreme Effort.** `iTaK Torch` inference would need a custom Tree-Attention decoder. The training loop would need to freeze the base `.gguf` and calculate gradients exclusively for the newly allocated output vectors.
- **What it takes (Dashboard):** A new mode dropdown `--mode medusa` and a slider for `Num Predict Heads`.

#### 3. ORPO & SimPO (Reference-Free Preference Optimization)

- **What it is:** Bleeding edge loss-functions that unify SFT and human-preference alignment without needing a massively heavy Reference Model.
- **What it takes (Go Engine):** **Low Effort.** These are purely mathematical loss function changes. We just add `case "orpo":` to `train_orchestrator.go` and calculate the odds-ratio mathematically against your existing Cross-Entropy logic.
- **What it takes (Dashboard):** A dropdown selector inside the UI mapping to `--mode orpo`.

#### 4. MoE Upcycling (Dense-to-MoE Expansion)

- **What it is:** Taking a standard model (like Qwen 8B) and cloning its Feed Forward layers into separate "Experts" controlled by a routing gate (similar to Kimi MoEs).
- **What it takes (Go Engine):** **High Effort.** Requires complex Go code physically restructuring the GGML Feed Forward layer formats to support custom Expert Routing kernels.
- **What it takes (Dashboard):** A new routing slider matrix UI mapping expert parameters.

### Legacy / Established Methods

#### 1. SFT (Supervised Fine-Tuning) & Standard LoRA

- **Status:** **ALREADY SUPPORTED NATIVELY.**
- **Integration:** The Dashboard literally just maps HTML sliders out to your existing baseline code flags.

#### 2. GRPO & Test-Time Compute (R1 / DeepSeek)

- **Status:** **ALREADY SUPPORTED NATIVELY.**
- **Integration:** You already wrote `runGRPOTraining()`. The Dashboard just needs a frontend text-area letting users custom-code their own deterministic Python/JSON reward evaluations.

#### 3. DPO (Direct Preference Optimization)

- **What it is:** The 2024 king of human alignment, requiring a "Chosen" and "Rejected" user feedback dataset.
- **What it takes (Go Engine):** **Medium Effort.** We update the JSONL dataset parser to accept chosen/rejected columns, and write the standard Bradley-Terry log-sigmoid loss algorithm into Go.
- **What it takes (Dashboard):** Data uploader mapped to Chosen/Rejected visual schema mapping.

#### 4. Old-School RLHF (PPO)

- **What it is:** The legacy 2023 ChatGPT training method.
- **What it takes (Go Engine):** **Not Recommended.** Extremely heavyweight. Requires booting up a massive secondary Reward Model tightly coupled in VRAM strictly to judge the outputs of the primary model. Your existing GRPO completely obliterates the need for this.
- **What it takes (Dashboard):** Complex parallel UI bindings to memory manage dual models.
