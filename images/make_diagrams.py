"""
Generate every README diagram as a Mermaid .mmd file, then render to PNG with the hand-drawn
look. Reproducible: edit a graph below, re-run, re-render.

Render step (run from this folder, needs Node.js for npx):
    for f in *.mmd; do
      npx -y @mermaid-js/mermaid-cli -i "$f" -o "${f%.mmd}.png" -c mmdc.json -p puppeteer.json -b white -s 3
    done
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Shared palette: soft fill + strong border per role (data, store, proc, model, rl, loss, ckpt, eval, out, op).
PALETTE = """
  classDef data fill:#cdeccd,stroke:#2e7d32,stroke-width:2px,color:#143d1a;
  classDef store fill:#cfeee9,stroke:#1f9e8f,stroke-width:2px,color:#0c3a34;
  classDef proc fill:#cfe3fb,stroke:#1565c0,stroke-width:2px,color:#0d2c52;
  classDef model fill:#ffe8a3,stroke:#d48806,stroke-width:2px,color:#5a3d00;
  classDef rl fill:#ffd9b3,stroke:#e67e22,stroke-width:2px,color:#6b3500;
  classDef loss fill:#ffd4d4,stroke:#c0392b,stroke-width:2px,color:#5c1212;
  classDef ckpt fill:#ececec,stroke:#666666,stroke-width:2px,color:#222222;
  classDef eval fill:#e7d6fb,stroke:#8e44ad,stroke-width:2px,color:#3d1a5a;
  classDef out fill:#e7d6fb,stroke:#7b3fbf,stroke-width:2px,color:#3d1a5a;
  classDef op fill:#ffffff,stroke:#888888,stroke-width:2px,color:#333333;
"""

DIAGRAMS = {
"00_pipeline": """flowchart LR
  PILE([The Pile<br/>raw text]):::data --> PRE{{Pretrain<br/>base ~400M}}:::model
  PRE --> SFT{{SFT<br/>instruct}}:::model
  SFT --> RM{{Reward Model<br/>Bradley-Terry}}:::rl
  SFT --> DPO{{DPO / ORPO / KTO}}:::rl
  SFT --> GRPO{{GRPO / RLVR}}:::rl
  RM -->|reward signal| PPO{{PPO}}:::rl
  PPO --> EVAL([GSM8K eval<br/>+ chat]):::eval
  DPO --> EVAL
  GRPO --> EVAL
""",

"01_data": """flowchart LR
  A1([Pretrain: The Pile]):::data --> A2[tiktoken r50k_base<br/>stream + encode]:::proc --> A3[(pile_train.h5<br/>flat int32 tokens)]:::store
  B1([SFT: Alpaca · Dolly · GSM8K]):::data --> B2[chat template<br/>+ mask the prompt]:::proc --> B3[(sft_packed.h5<br/>tokens + loss_mask)]:::store
  C1([Prefs: HH-RLHF · UltraFeedback]):::data --> C2[split prompt /<br/>chosen / rejected]:::proc --> C3[(preferences.jsonl)]:::store
  D1([RL: GSM8K · arithmetic]):::data --> D2[keep the gold answer]:::proc --> D3[(rl_prompts.jsonl)]:::store
""",

"02_tokenization": """flowchart LR
  T1["raw text<br/>'Once upon a time'"]:::data --> T2["tiktoken<br/>r50k_base"]:::proc
  T2 --> T3["token ids<br/>[7454, 2402, 257, 640]"]:::store
  T3 --> T4["append &lt;|endoftext|&gt;<br/>(id 50256)"]:::proc
  T4 --> T5["fixed windows of<br/>context_length + 1"]:::store
  T5 --> T6["x = window[:-1]<br/>y = window[1:]"]:::out
""",

"03_mlp": """flowchart LR
  M1["x  (B, T, n_embed)"]:::data --> M2["Linear<br/>n_embed -> 4 · n_embed"]:::proc
  M2 --> M3["ReLU"]:::proc
  M3 --> M4["Linear<br/>4 · n_embed -> n_embed"]:::proc
  M4 --> M5["out  (B, T, n_embed)"]:::out
""",

"04_attention_head": """flowchart LR
  X["x : token vectors<br/>(B, T, C)"]:::data --> Q["Q = x Wq"]:::proc
  X --> K["K = x Wk"]:::proc
  X --> V["V = x Wv"]:::proc
  Q --> S["scores = Q Kᵀ / √d"]:::rl
  K --> S
  S --> M["causal mask<br/>+ softmax"]:::rl
  M --> O["weights · V"]:::rl
  V --> O
  O --> H["head output<br/>(B, T, head_size)"]:::out
""",

"05_multi_head_attention": """flowchart LR
  X["x  (B, T, C)"]:::data --> H1["head 1"]:::proc
  X --> H2["head 2"]:::proc
  X --> H3["...  (n_head heads)"]:::proc
  H1 --> CC["concat heads<br/>(B, T, C)"]:::rl
  H2 --> CC
  H3 --> CC
  CC --> PJ["output projection<br/>Linear C -> C"]:::proc
  PJ --> HO["multi-head output"]:::out
""",

"06_transformer_block": """flowchart LR
  X["x"]:::data --> LN1["LayerNorm"]:::proc --> AT["Multi-Head Attention"]:::rl --> A1(("+")):::op
  X --> A1
  A1 --> LN2["LayerNorm"]:::proc --> ML["MLP"]:::proc --> A2(("+")):::op
  A1 --> A2
  A2 --> BO["block output"]:::out
""",

"07_transformer": """flowchart LR
  I["token ids  (B, T)"]:::data --> TE["token embedding"]:::proc
  I --> PE["position embedding"]:::proc
  TE --> SUM(("+")):::op
  PE --> SUM
  SUM --> BLK["N_BLOCKS ×<br/>Transformer Block"]:::model
  BLK --> LNF["final LayerNorm"]:::proc
  LNF --> LMH["lm_head<br/>Linear -> vocab_size"]:::proc
  LMH --> LOG["logits  (B, T, vocab)"]:::out
  LOG -. targets present .-> CE["cross-entropy loss"]:::loss
""",

"08_training_loop": """flowchart LR
  D[(pile_train.h5)]:::store --> IT["get_batch_iterator<br/>random windows"]:::proc --> FW["forward<br/>(bf16 autocast)"]:::model --> CE["cross-entropy"]:::loss --> BW["backward<br/>× grad_accum"]:::proc --> CL["clip grad norm 1.0"]:::proc --> ST["AdamW step<br/>cosine LR + warmup"]:::model --> CK[(checkpoint)]:::ckpt
  ST -. next step .-> IT
""",

"09_sft": """flowchart LR
  S0[(sft_packed.h5<br/>tokens + loss_mask)]:::store --> S1["Transformer forward"]:::model --> S2["shift: predict t+1"]:::proc --> S3["per-token cross-entropy"]:::loss
  S0 --> S4["loss_mask = 1 only on<br/>assistant tokens"]:::proc --> S5["mean over masked tokens"]:::loss
  S3 --> S5 --> S6["AdamW step"]:::model
""",

"10_reward_model": """flowchart LR
  R0["prompt + chosen / rejected"]:::data --> R1["SFT backbone<br/>forward_hidden"]:::model --> R2["take last real token"]:::proc --> R3["reward head<br/>Linear -> 1"]:::proc
  R3 --> R4["r_chosen , r_rejected"]:::rl --> R5["Bradley-Terry loss<br/>-log σ(r_chosen - r_rejected)"]:::loss --> R6["AdamW step"]:::model
""",

"11_dpo": """flowchart LR
  P0["chosen / rejected pair"]:::data --> P1["policy (trainable)<br/>sequence log-probs"]:::model
  P0 --> P2["reference (frozen SFT)<br/>sequence log-probs"]:::ckpt
  P1 --> P3["DPO loss<br/>-log σ(β · Δ)"]:::loss
  P2 --> P3 --> P4["AdamW step"]:::model
""",

"12_ppo": """flowchart LR
  Q0([GSM8K prompts]):::data --> Q1["rollout<br/>generate + log-probs"]:::model --> Q2["score<br/>verifier or reward model"]:::rl --> Q3["+ per-token KL to ref"]:::rl --> Q4["GAE<br/>advantages + returns"]:::proc --> Q5["clipped policy + value<br/>(K epochs)"]:::model --> Q6[(ppo.pt)]:::ckpt
  Q5 -. sync old policy .-> Q1
""",

"13_grpo": """flowchart LR
  G0([prompt]):::data --> G1["sample a GROUP<br/>of G answers"]:::model --> G2["verifier reward<br/>per answer"]:::rl --> G3["group advantage<br/>(r - mean) / std"]:::proc --> G4["clipped surrogate<br/>+ k3 KL to ref"]:::loss --> G5["policy update"]:::model
  G5 -. next prompt .-> G0
""",

"14_evaluation": """flowchart LR
  E0[(stage checkpoint)]:::ckpt --> E1["greedy generate"]:::model --> E2["extract_answer<br/>&lt;answer&gt;N&lt;/answer&gt;"]:::proc --> E3{"== gold?"}:::rl --> E4["GSM8K accuracy<br/>Base -> SFT -> ... -> GRPO"]:::eval
""",

"15_inference": """flowchart LR
  C0[(any checkpoint)]:::ckpt --> C1["load dims from cfg"]:::proc --> C2{"chat or raw?"}:::rl
  C2 -->|instruct| C3["wrap in chat template"]:::proc
  C2 -->|base| C4["raw prefix"]:::proc
  C3 --> C5["generate<br/>temperature / top-p / greedy"]:::model
  C4 --> C5 --> C6["decode -> reply"]:::out
""",
}

def main():
    for name, body in DIAGRAMS.items():
        path = os.path.join(OUT, name + ".mmd")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body.rstrip() + "\n" + PALETTE)
        print("wrote", name + ".mmd")
    print("total:", len(DIAGRAMS), "diagrams")

if __name__ == "__main__":
    main()
