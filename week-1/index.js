import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { ChatMistralAI } from "@langchain/mistralai";
import { config } from "dotenv";

config();

class LLMResponse {
  constructor(provider, model, prompt, response, latencyMs, tokensUsed = null, error = null) {
    this.provider = provider;
    this.model = model;
    this.prompt = prompt;
    this.response = response;
    this.latencyMs = latencyMs;
    this.tokensUsed = tokensUsed;
    this.error = error;
  }
}

class LLMProvider {
  constructor(apiKey, model, name) {
    this.apiKey = apiKey;
    this.model = model;
    this.name = name;
  }

  createModel() {
    throw new Error("createModel() must be implemented");
  }

  async generate(prompt) {
    const model = this.createModel();
    try {
      const start = Date.now();
      const aiMessage = await model.invoke(prompt);
      const latency = Date.now() - start;

      const usage = aiMessage.usage_metadata;
      const tokensUsed = usage
        ? usage.total_tokens || (usage.input_tokens + usage.output_tokens)
        : null;

      return new LLMResponse(
        this.name,
        this.model,
        prompt,
        aiMessage.content,
        latency,
        tokensUsed
      );
    } catch (error) {
      return new LLMResponse(this.name, this.model, prompt, "", 0, null, error.message);
    }
  }
}

class OpenAIProvider extends LLMProvider {
  constructor(apiKey, model) {
    super(apiKey, model, "OpenAI");
  }

  createModel() { 
    return new ChatOpenAI({ apiKey: this.apiKey, model: this.model, });
  }
}

class AnthropicProvider extends LLMProvider {
  constructor(apiKey, model) {
    super(apiKey, model, "Anthropic");
  }

  createModel() {
    return new ChatAnthropic({
      apiKey: this.apiKey,
      model: this.model,
      maxTokens: 1000,
    });
  }
}

class GeminiProvider extends LLMProvider {
  constructor(apiKey, model) {
    super(apiKey, model, "Google Gemini");
  }

  createModel() {
    return new ChatGoogleGenerativeAI({ apiKey: this.apiKey, model: this.model });
  }
}

class GroqProvider extends LLMProvider {
  constructor(apiKey, model) {
    super(apiKey, model, "Groq");
  }

  createModel() {
    return new ChatGroq({ apiKey: this.apiKey, model: this.model });
  }
}

class MistralProvider extends LLMProvider {
  constructor(apiKey, model) {
    super(apiKey, model, "Mistral");
  }

  createModel() {
    return new ChatMistralAI({ apiKey: this.apiKey, model: this.model, temperature: 1,'top-p' : 1,maxTokens: 800,});
  }
}

const PROVIDER_CONFIGS = {
  openai: {
    class: OpenAIProvider,
    model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
  },
  anthropic: {
    class: AnthropicProvider,
    model: process.env.ANTHROPIC_MODEL || "claude-3-5-sonnet-20241022",
    apiKey: process.env.ANTHROPIC_API_KEY,
  },
  gemini: {
    class: GeminiProvider,
    model: process.env.GEMINI_MODEL || "gemini-2.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
  },
  groq: {
    class: GroqProvider,
    model: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
    apiKey: process.env.GROQ_API_KEY,
  },
  mistral: {
    class: MistralProvider,
    model: process.env.MISTRAL_MODEL || "mistral-large-latest",
    apiKey: process.env.MISTRAL_API_KEY,
  },
};

function getProvider(name) {
  const config = PROVIDER_CONFIGS[name];
  if (!config || !config.apiKey) return null;
  return new config.class(config.apiKey, config.model);
}

async function runComparison(prompts, providers = null) {
  if (!providers) {
    providers = Object.keys(PROVIDER_CONFIGS).filter(p => PROVIDER_CONFIGS[p].apiKey);
  }

  const results = [];
  for (const providerName of providers) {
    const provider = getProvider(providerName);
    if (!provider) {
      console.log(`Skipping ${providerName}: not configured`);
      continue;
    }

    console.log(`\nTesting ${provider.name} (${provider.model})...`);
    for (const prompt of prompts) {
      console.log(`  Prompt: ${prompt.slice(0, 50)}...`);
      const result = await provider.generate(prompt);
      results.push(result);
      if (result.error) {
        console.log(`    Error: ${result.error}`);
      } else {
        console.log(`    Latency: ${result.latencyMs}ms, Tokens: ${result.tokensUsed || "N/A"}`);
      }
    }
  }

  return results;
}

function printComparisonTable(results) {
  console.log("\n" + "=".repeat(100));
  console.log("COMPARISON RESULTS");
  console.log("=".repeat(100));

  const byPrompt = {};
  for (const r of results) {
    if (!byPrompt[r.prompt]) byPrompt[r.prompt] = [];
    byPrompt[r.prompt].push(r);
  }

  for (const [prompt, responses] of Object.entries(byPrompt)) {
    console.log(`\nPrompt: ${prompt}`);
    console.log("-".repeat(80));
    console.log(
      `${"Provider".padEnd(20)} ${"Model".padEnd(30)} ${"Latency (ms)".padEnd(15)} ${"Tokens".padEnd(10)} Status`
    );
    console.log("-".repeat(80));
    for (const r of responses) {
      const status = r.error ? `ERROR: ${String(r.error).slice(0, 30)}` : "OK";
      console.log(
        `${r.provider.padEnd(20)} ${r.model.padEnd(30)} ${String(r.latencyMs).padEnd(15)} ${String(r.tokensUsed || "N/A").padEnd(10)} ${status}`
      );
    }
  }
}

function saveResults(results, filename = "comparison_results.json") {
  import("fs").then(fs => {
    fs.writeFileSync(filename, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to ${filename}`);
  });
}

export { runComparison, printComparisonTable, saveResults, PROVIDER_CONFIGS };