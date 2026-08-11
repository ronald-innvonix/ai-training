# LLM Provider Benchmark Report

Comparison of response quality, latency, and token usage across three providers — **Google Gemini (gemini-3.6-flash)**, **Groq (llama-3.3-70b-versatile)**, and **Mistral (mistral-large-latest)** — on five standard prompts, plus a separate creative-writing experiment testing temperature/top-p settings.

---

## 1. Standard Prompt Set

**Settings:** `temperature: 0`, `top-p: 0.5`, `maxTokens: 100` _(unless otherwise noted — see performance table)_

### 1.1 Prompt: "Explain quantum computing in simple terms."

| Provider      | Model                   | Latency (ms) | Tokens Used | Notes                                                          |
| ------------- | ----------------------- | ------------ | ----------- | -------------------------------------------------------------- |
| Google Gemini | gemini-3.6-flash        | 9,698        | 1,725       | Coin/maze analogies, structured with headers, covers use cases |
| Groq          | llama-3.3-70b-versatile | 1,282        | 491         | Concise, coin analogy, fastest response                        |
| Mistral       | mistral-large-latest    | 17,159       | 923         | Most detailed — includes limitations/caveats section           |

### 1.2 Prompt: "Write a short story about a robot learning to paint."

| Provider      | Model                   | Latency (ms) | Tokens Used | Notes                                                             |
| ------------- | ----------------------- | ------------ | ----------- | ----------------------------------------------------------------- |
| Google Gemini | gemini-3.6-flash        | 21,510       | 2,924       | Longest, most literary; strong narrative arc (Master Elias/Seven) |
| Groq          | llama-3.3-70b-versatile | 2,884        | 666         | Shorter narrative (Dr. Kim/Zeta), competent but generic           |
| Mistral       | mistral-large-latest    | 10,878       | 526         | Sci-fi setting (abandoned colony), atmospheric, concise           |

### 1.3 Prompt: "Debug this code: `for (let i = 0; i < 10; i++) { console.log(i); if (i === 5) break; }`"

| Provider      | Model                   | Latency (ms) | Tokens Used | Notes                                          |
| ------------- | ----------------------- | ------------ | ----------- | ---------------------------------------------- |
| Google Gemini | gemini-3.6-flash        | 4,911        | 971         | Correctly identifies no bug; offers 3 variants |
| Groq          | llama-3.3-70b-versatile | 1,551        | 535         | Correct; slightly verbose step-by-step framing |
| Mistral       | mistral-large-latest    | 3,775        | 315         | Correct; concise, offers 3 variants            |

### 1.4 Prompt: "Write a SQL query to find the second highest salary from employees table."

| Provider      | Model                   | Latency (ms) | Tokens Used | Notes                                                                |
| ------------- | ----------------------- | ------------ | ----------- | -------------------------------------------------------------------- |
| Google Gemini | gemini-3.6-flash        | 6,909        | 1,383       | 4 methods incl. `DENSE_RANK`, `OFFSET/FETCH`, recommendation summary |
| Groq          | llama-3.3-70b-versatile | 665          | 261         | 2 methods, fastest and most concise                                  |
| Mistral       | mistral-large-latest    | 4,868        | 374         | 5 methods incl. `ROW_NUMBER`, `TOP`, NULL-handling                   |

### 1.5 Prompt: "Explain the difference between async/await and Promise.then in JavaScript."

| Provider      | Model                   | Latency (ms) | Tokens Used | Notes                                                            |
| ------------- | ----------------------- | ------------ | ----------- | ---------------------------------------------------------------- |
| Google Gemini | gemini-3.6-flash        | 12,215       | 2,380       | Comparison table, scope/debugging deep dive, best-practice tip   |
| Groq          | llama-3.3-70b-versatile | 437          | 147         | Code-only answer, no prose explanation — fastest, least thorough |
| Mistral       | mistral-large-latest    | 20,706       | 1,300       | Most exhaustive — 6 comparison dimensions + recommendation table |

---

## 2. Performance Summary

| Provider      | Avg Latency (ms) | Avg Tokens Used | Characteristic Style                                    |
| ------------- | ---------------- | --------------- | ------------------------------------------------------- |
| Google Gemini | 11,049           | 1,877           | Long-form, heavily structured with headers/analogies    |
| Groq          | 1,364            | 420             | Fastest by a wide margin; terse, sometimes skips prose  |
| Mistral       | 11,477           | 688             | Most exhaustive coverage of edge cases and alternatives |

**Key takeaways:**

- **Groq** is consistently the fastest (sub-3s for 4/5 prompts) but gives the thinnest answers — notably skipping explanatory prose entirely on the async/await prompt.
- **Google Gemini** produces the longest, most narratively rich creative writing and the most polished explanatory formatting.
- **Mistral** tends toward the most technically exhaustive answers (most SQL methods, most async/await comparison angles) at the cost of higher latency.

---

## 3. Temperature / Top-P Experiment (Mistral Only)

**Prompt:** _"Imagine a historical event called the 'Great Moonlight War of 1847.' Explain what caused it, which countries participated, and what the outcome was."_

This fictional-history prompt was run at three settings to observe how temperature and top-p affect creativity, coherence, and output length.

| Run | Temperature | Top-P | Max Tokens | Latency (ms) | Tokens Used |
| --- | ----------- | ----- | ---------- | ------------ | ----------- |
| 1   | 0           | 0.5   | 100        | 3,649        | 136         |
| 2   | 0.5         | 0.9   | 600        | 15,623       | 636         |
| 3   | 1           | 1     | 800        | 16,349       | 836         |

### Observations

- **Run 1 (temp 0, top-p 0.5):** Truncated almost immediately due to the low `maxTokens` cap. Sets up a premise (scientific rivalry, colonial ambition, a "Lunar Mirage") but cuts off mid-sentence.
- **Run 2 (temp 0.5, top-p 0.9):** Fuller narrative — introduces a "Lunar Anomaly of 1846," a fictional mineral ("Seleneite"), and named alliances (Franco-Prussian Lunar Alliance). Still cuts off mid-scene at the "Moonlight Incident."
- **Run 3 (temp 1, top-p 1):** Most elaborate and whimsical version — introduces the "Moonlight Doctrine," named historical-adjacent figures (Friedrich Wilhelm IV, Louis-Philippe), and an unusual coalition list including Texas and the Ottoman Empire under a "Lunar League" vs. "Solar Coalition" framing. Highest internal consistency and worldbuilding detail, though also cut off before reaching a resolution.

**Trend:** Increasing temperature and top-p (alongside larger token budgets) produced progressively more elaborate, internally consistent, and imaginative alternate-history detail — richer named entities, causal chains, and factional structure — at the cost of higher latency and no run reaching a full resolution within its token budget.

---
