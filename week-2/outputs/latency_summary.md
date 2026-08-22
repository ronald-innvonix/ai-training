# Latency Comparison Across Providers

## Per-method average latency (across providers)
Method    Avg total (s)   Avg first-token (s) Avg chars 
invoke    2.676           -                   170.3     
stream    1.964           1.518               149.7     
ainvoke   3.079           -                   157.7     
astream   2.080           1.655               149.7     

## Provider ranking by average total latency
- ChatGroq: 0.621 s (avg)
- ChatMistralAI: 1.905 s (avg)
- ChatGoogleGenerativeAI: 4.823 s (avg)

## Findings
Streaming methods (stream / astream) return their *first token* earlier than the total latency of invoke/ainvoke, so they feel more responsive even when total completion time is similar.
Async vs sync single-call latency: ainvoke/invoke ratio = 0.91x (values close to 1x indicate async has ~no per-call advantage for a single request; its real benefit shows under concurrent workloads).

Caveat: measured on one request per method against cloud APIs; latency varies with provider load, region and time of day.