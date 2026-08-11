import { runComparison, printComparisonTable, saveResults, PROVIDER_CONFIGS } from "./index.js";

const testPrompts = [
  // "Explain quantum computing in simple terms.",
  // "Write a short story about a robot learning to paint.",
  // "Debug this code: for (let i = 0; i < 10; i++) { console.log(i); if (i === 5) break; }",
  // "Write a SQL query to find the second highest salary from employees table.",
  // "Explain the difference between async/await and Promise.then in JavaScript.",
  "Imagine a historical event called the ‘Great Moonlight War of 1847.’ Explain what caused it, which countries participated, and what the outcome was."
];

async function main() {
  console.log("LLM Provider Comparison Tool (Node.js)");
  console.log("=".repeat(50));

  const available = Object.keys(PROVIDER_CONFIGS).filter(p => PROVIDER_CONFIGS[p].apiKey);
  console.log(`Configured providers: ${available.length > 0 ? available.join(", ") : "None"}`);

  if (available.length === 0) {
    console.log("\nPlease configure API keys in .env file (see .env.example)");
    process.exit(1);
  }

  const results = await runComparison(testPrompts, available);
  printComparisonTable(results);
  saveResults(results);
}

main().catch(console.error);