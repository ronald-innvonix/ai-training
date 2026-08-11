import Groq from "groq-sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "dotenv";
config();

async function listGemini() {
    try {
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        // there is no easy listModels in this old SDK version, but wait it might have one or I can grab it from api
        // let's do a curl to check api
    } catch(e) {}
}

async function listGroq() {
    try {
        const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
        const models = await groq.models.list();
        console.log("Groq Models:", models.data.map(m => m.id));
    } catch(e) { console.error(e) }
}
listGroq();

