"""
Day 13: Job Description Analyzer Agent

A capstone project that integrates Days 1-12 into a practical AI agent:
- Token counting (Day 1)
- Context management (Day 2)
- Prompt caching (Day 3)
- Model/provider abstraction (Day 4, Day 12)
- Code review patterns (Day 5)
- Transformer concepts (Days 6-7)
- VRAM calculation (Day 8)
- Agent loop patterns (Day 9, Day 11)
- Structured outputs (Day 10, Day 12)

This agent:
1. Parses a job description using structured outputs
2. Extracts required skills and technologies
3. Compares against user's current skills
4. Generates a personalized learning roadmap
5. Uses tools to estimate learning costs/time

Run: python experiments/13_job_analyzer_agent.py
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# --- Data Models ---

@dataclass
class Skill:
    """Represents a technical skill."""
    name: str
    category: str  # "technical", "framework", "concept", "tool"
    proficiency: int  # 1-5 scale: 1=beginner, 3=intermediate, 5=expert
    days_to_learn: int  # Estimated days to reach proficiency 3
    
    def __repr__(self):
        return f"{self.name} ({self.category}, level: {self.proficiency})"


@dataclass
class JobDescription:
    """Parsed job description with extracted requirements."""
    title: str
    company: str
    location: str
    required_skills: List[Dict[str, Any]]
    preferred_skills: List[Dict[str, Any]]
    experience_years: str
    raw_text: str
    

@dataclass
class LearningPlan:
    """Personalized learning plan based on skill gaps."""
    skills_to_learn: List[Skill]
    total_estimated_hours: int
    timeline_weeks: int
    priority_order: List[str]  # Skill names ordered by importance
    resources: Dict[str, List[str]]  # Skill -> list of resource URLs/names
    confidence: float


@dataclass
class LearningRecommendation:
    """Complete AI learning recommendation."""
    summary: str
    skill_gaps: List[Dict[str, Any]]
    learning_path: LearningPlan
    estimated_cost: float
    estimated_time: str
    hardware_recommendations: List[str]
    project_ideas: List[str]


# --- Sample Data ---

SAMPLE_JOB_DESCRIPTION = """
AI Engineer - Machine Learning

Company: TechVision Inc.
Location: Remote

Job Description:
We're seeking an AI Engineer to build and deploy machine learning systems. The ideal candidate has:

Required Skills:
- Proficiency in Python and PyTorch
- Understanding of transformer architectures (attention mechanism, tokenization)
- Experience with model quantization and optimization
- Knowledge of Hugging Face ecosystem
- Familiarity with LLM APIs (OpenAI, Anthropic)
- Experience with prompt engineering and prompt caching
- Understanding of VRAM optimization and hardware constraints
- Basic knowledge of agent frameworks

Preferred Skills:
- Experience with fine-tuning (LoRA, QLoRA)
- Knowledge of RAG (Retrieval-Augmented Generation)
- Experience building AI agents
- Familiarity with cloud platforms (AWS SageMaker, GCP Vertex AI)
- Docker and CI/CD for ML deployment

Experience: 2+ years hands-on ML engineering experience

Responsibilities:
- Build and deploy LLM-powered applications
- Optimize models for cost and latency
- Implement prompt caching strategies
- Design agent workflows for automation
- Collaborate with engineering teams
"""

# Your current skills (based on Days 1-12 learning)
CURRENT_SKILLS = [
    Skill("Python", "language", 4, 1),
    Skill("Tokenization", "concept", 3, 3),
    Skill("Attention Mechanism", "concept", 3, 5),
    Skill("Transformer Architecture", "concept", 2, 7),
    Skill("Layer Normalization", "concept", 2, 3),
    Skill("Feed-Forward Networks", "concept", 2, 3),
    Skill("VRAM Calculation", "concept", 3, 3),
    Skill("Quantization (INT4/INT8)", "concept", 2, 5),
    Skill("Prompt Caching", "concept", 3, 5),
    Skill("Structured Outputs", "concept", 3, 3),
    Skill("Agent Loops", "concept", 3, 7),
    Skill("API Integration", "tool", 3, 5),
    Skill("Provider Abstraction", "tool", 2, 5),
    Skill("Error Handling", "tool", 3, 2),
    Skill("Cost Tracking", "tool", 3, 2),
    Skill("OpenAI API", "tool", 1, 3),
    Skill("Anthropic API", "tool", 1, 5),
    Skill("Hugging Face", "framework", 1, 7),
    Skill("PyTorch", "framework", 1, 10),
    Skill("LLM APIs", "tool", 1, 7),
    Skill("Prompt Engineering", "concept", 2, 7),
    Skill("Model Optimization", "concept", 1, 10),
    Skill("Fine-tuning (LoRA)", "concept", 1, 10),
    Skill("RAG", "concept", 1, 8),
    Skill("Agent Frameworks", "framework", 1, 5),
    Skill("Docker", "tool", 1, 5),
    Skill("CI/CD", "tool", 1, 5),
    Skill("Cloud ML Platforms", "tool", 1, 8),
]


# --- Structured Output Schemas ---

JOB_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "priority": {"type": "string", "enum": ["required", "preferred"]}
                }
            }
        },
        "experience_years": {"type": "string"},
    },
    "required": ["title", "skills", "experience_years"]
}

LEARNING_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "skill_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "current_level": {"type": "integer"},
                    "target_level": {"type": "integer"},
                    "gap": {"type": "integer"},
                    "estimated_hours": {"type": "integer"}
                }
            }
        },
        "learning_path": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "week": {"type": "integer"},
                            "focus": {"type": "string"},
                            "hours": {"type": "integer"},
                            "resources": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "total_weeks": {"type": "integer"},
                "total_hours": {"type": "integer"}
            }
        },
        "hardware_recommendations": {"type": "array", "items": {"type": "string"}},
        "project_ideas": {"type": "array", "items": {"type": "string"}},
        "estimated_cost": {"type": "number"},
        "estimated_time": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["summary", "skill_gaps", "learning_path", "hardware_recommendations", "project_ideas", "estimated_cost", "estimated_time", "confidence"]
}


# --- Tool Functions (Integrated from Days 1-12) ---

def calculate_vram(params: float, dtype: str, seq_len: int = 2048) -> Dict:
    """Calculate VRAM requirements (from Day 8)."""
    bytes_per = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}[dtype]
    weights = (params * bytes_per) / (1024 ** 3)
    kv_cache = (2 * 32 * 4096 * seq_len * bytes_per) / (1024 ** 3)
    total = weights + kv_cache + weights * 0.15 + (weights + kv_cache) * 0.1
    return {"weights_gb": round(weights, 2), "kv_cache_gb": round(kv_cache, 2), "total_gb": round(total, 2)}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (from Day 1)."""
    return len(text) // 4  # Rough: ~4 chars per token for English


def find_compatible_models(vram_gb: float) -> List[str]:
    """Find models that fit hardware (from Day 4, Day 8)."""
    models = {
        "Qwen3.5 4B (INT4)": 2.6, "Phi-3 Mini (INT4)": 1.6, "Qwen3.5 0.6B (INT4)": 0.9,
        "Qwen3.5 8B (INT4)": 4.8, "Gemma 2B (INT4)": 1.2
    }
    return [name for name, vram in models.items() if vram <= vram_gb]


def calculate_cost(usage_tokens: int, provider: str = "anthropic") -> float:
    """Estimate cost for token usage (from Day 3, Day 12)."""
    cost_per_1k = {"anthropic": 0.003, "openai": 0.0025, "ollama": 0.0}
    return (usage_tokens / 1000) * cost_per_1k.get(provider, 0.003)


# --- Simulated LLM Response ---

def simulate_agent_response(prompt: str, schema: dict) -> dict:
    """Simulates an LLM structured response."""
    if "job" in prompt.lower() and "title" in prompt.lower():
        return {
            "title": "AI Engineer",
            "skills": [
                {"name": "Python", "category": "language", "priority": "required"},
                {"name": "PyTorch", "category": "framework", "priority": "required"},
                {"name": "Transformer Architecture", "category": "concept", "priority": "required"},
                {"name": "Attention Mechanism", "category": "concept", "priority": "required"},
                {"name": "Hugging Face", "category": "framework", "priority": "required"},
                {"name": "Quantization", "category": "concept", "priority": "required"},
                {"name": "VRAM Optimization", "category": "concept", "priority": "required"},
                {"name": "LLM APIs", "category": "tool", "priority": "required"},
                {"name": "Prompt Engineering", "category": "concept", "priority": "required"},
                {"name": "Agent Frameworks", "category": "framework", "priority": "required"},
                {"name": "Fine-tuning (LoRA)", "category": "concept", "priority": "preferred"},
                {"name": "RAG", "category": "concept", "priority": "preferred"},
                {"name": "Docker", "category": "tool", "priority": "preferred"},
            ],
            "experience_years": "2+ years"
        }
    elif "gap" in prompt.lower() or "path" in prompt.lower():
        return {
            "summary": "You need to deepen knowledge in PyTorch, Hugging Face, and fine-tuning to qualify for AI Engineer roles. Your foundation (transformers, attention, quantization) is strong — focus on hands-on application.",
            "skill_gaps": [
                {"skill": "PyTorch", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 40},
                {"skill": "Hugging Face", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 30},
                {"skill": "Fine-tuning (LoRA)", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 35},
                {"skill": "RAG", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 25},
                {"skill": "LLM APIs", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 20},
                {"skill": "Prompt Engineering", "current_level": 2, "target_level": 4, "gap": 2, "estimated_hours": 20},
                {"skill": "Agent Frameworks", "current_level": 1, "target_level": 3, "gap": 2, "estimated_hours": 25},
            ],
            "learning_path": {
                "steps": [
                    {
                        "week": 1,
                        "focus": "PyTorch fundamentals and tensor operations",
                        "hours": 15,
                        "resources": [
                            "PyTorch official tutorials (pytorch.org/tutorials)",
                            "Hands-on ML with PyTorch book",
                            "Build simple neural network from scratch"
                        ]
                    },
                    {
                        "week": 2,
                        "focus": "Hugging Face ecosystem and model loading",
                        "hours": 12,
                        "resources": [
                            "Hugging Face course",
                            "Transformers library documentation",
                            "Fine-tune a pre-trained model"
                        ]
                    },
                    {
                        "week": 3,
                        "focus": "Fine-tuning with LoRA and QLoRA",
                        "hours": 15,
                        "resources": [
                            "PEFT library documentation",
                            "LoRA paper tutorial",
                            "Fine-tune LLaMA on custom data"
                        ]
                    },
                    {
                        "week": 4,
                        "focus": "RAG and agent frameworks",
                        "hours": 10,
                        "resources": [
                            "LangChain RAG tutorial",
                            "OpenAI Agents SDK",
                            "Build a simple RAG chatbot"
                        ]
                    }
                ],
                "total_weeks": 4,
                "total_hours": 52
            },
            "hardware_recommendations": [
                "Your GTX 1650 (4GB) is sufficient for INT4 quantized 4B models",
                "Consider upgrading to RTX 4060/4070 (8GB) for larger models",
                "Use cloud GPUs for fine-tuning experiments"
            ],
            "project_ideas": [
                "Build a VRAM calculator web app (integrate your Day 8 work)",
                "Create an agent that explains ML concepts using local LLMs",
                "Fine-tune Qwen on your learning notes to create a study assistant"
            ],
            "estimated_cost": 25.0,
            "estimated_time": "4 weeks, ~15 hours/week",
            "confidence": 0.92
        }
    else:
        return {
            "summary": "General AI learning recommendation",
            "skill_gaps": [],
            "learning_path": {"steps": [], "total_weeks": 0, "total_hours": 0},
            "hardware_recommendations": [],
            "project_ideas": [],
            "estimated_cost": 0.0,
            "estimated_time": "N/A",
            "confidence": 0.5
        }


# --- Main Agent Class ---

class JobAnalyzerAgent:
    """
    AI Agent that analyzes job descriptions and generates learning roadmaps.
    """

    def __init__(self, current_skills: List[Skill], hardware_profile: Dict):
        self.current_skills = {s.name.lower(): s for s in current_skills}
        self.hardware = hardware_profile
        self.conversation_log: List[Dict] = []

    def analyze_job(self, job_text: str) -> JobDescription:
        """Step 1: Parse the job description."""
        prompt = f"""
        Parse this job description into structured JSON:
        {job_text}

        Extract: title, required and preferred skills with categories, experience years.
        Use schema: {json.dumps(JOB_PARSE_SCHEMA)}
        """

        # Simulate LLM call
        result = simulate_agent_response(prompt, JOB_PARSE_SCHEMA)

        parsed = JobDescription(
            title=result["title"],
            company="TechVision Inc.",
            location="Remote",
            required_skills=[s for s in result["skills"] if s["priority"] == "required"],
            preferred_skills=[s for s in result["skills"] if s["priority"] == "preferred"],
            experience_years=result["experience_years"],
            raw_text=job_text
        )

        print(f"\nParsed Job: {parsed.title}")
        print(f"Required skills: {len(parsed.required_skills)}")
        print(f"Preferred skills: {len(parsed.preferred_skills)}")

        return parsed

    def compute_skill_gaps(self, job: JobDescription) -> List[Dict]:
        """Step 2: Compare job requirements against current skills."""
        gaps = []

        all_required = job.required_skills + job.preferred_skills

        for skill_entry in all_required:
            skill_name = skill_entry["name"].lower()
            found = False

            for current_name, current_skill in self.current_skills.items():
                if skill_name in current_name or current_name in skill_name:
                    found = True
                    current_level = current_skill.proficiency
                    target_level = 3  # Target: intermediate proficiency

                    # Map proficiency to estimated hours
                    hours_needed = current_skill.days_to_learn * (target_level - current_level) if target_level > current_level else 0

                    gaps.append({
                        "skill": skill_entry["name"],
                        "current_level": current_level,
                        "target_level": target_level,
                        "gap": max(0, target_level - current_level),
                        "estimated_hours": hours_needed if hours_needed > 0 else 15,
                    })
                    break

            if not found:
                gaps.append({
                    "skill": skill_entry["name"],
                    "current_level": 1,
                    "target_level": 3,
                    "gap": 2,
                    "estimated_hours": 20,
                })

        return gaps

    def generate_learning_path(self, gaps: List[Dict]) -> LearningPlan:
        """Step 3: Generate optimized learning path."""

        # Sort by gap priority (largest gaps first)
        sorted_gaps = sorted(gaps, key=lambda x: (x["gap"], x["estimated_hours"]), reverse=True)

        # Group into weekly blocks (~15 hours/week)
        hours_per_week = 15
        remaining_gaps = sorted_gaps.copy()
        weeks = []
        week_num = 1
        total_hours = sum(g["estimated_hours"] for g in gaps)

        while remaining_gaps:
            week_hours = 0
            week_gaps = []

            while remaining_gaps and week_hours < hours_per_week:
                gap = remaining_gaps.pop(0)
                gap_hours = gap["estimated_hours"]

                if week_hours + gap_hours <= hours_per_week:
                    # Fits in this week
                    week_gaps.append(gap)
                    week_hours += gap_hours
                elif gap_hours > hours_per_week:
                    # Gap is too big for one week - assign it alone
                    week_gaps.append(gap)
                    week_hours = gap_hours
                else:
                    # Doesn't fit but is smaller than a week - put it back for next week
                    remaining_gaps.append(gap)
                    break

            if week_gaps:
                focus = ", ".join(g["skill"] for g in week_gaps[:2])
                weeks.append({
                    "week": week_num,
                    "focus": focus if len(focus) < 50 else focus[:47] + "...",
                    "hours": week_hours,
                    "resources": self._get_resources(week_gaps),
                    "skills": [g["skill"] for g in week_gaps]
                })
                week_num += 1

        total_weeks = len(weeks)

        return LearningPlan(
            skills_to_learn=[Skill(g["skill"], "concept", g["target_level"], g["estimated_hours"]) for g in gaps],
            total_estimated_hours=total_hours,
            timeline_weeks=total_weeks,
            priority_order=[g["skill"] for g in sorted_gaps],
            resources={},
            confidence=0.9
        ), weeks

    def _get_resources(self, gaps: List[Dict]) -> Dict[str, List[str]]:
        """Generate resource recommendations for skills."""
        resources = {}
        for gap in gaps:
            skill = gap["skill"].lower()
            if "pytorch" in skill:
                resources["pytorch"] = [
                    "https://pytorch.org/tutorials/",
                    "Deep Learning with PyTorch book",
                    "Build a CNN from scratch"
                ]
            elif "hugging" in skill:
                resources["hugging_face"] = [
                    "https://huggingface.co/learn",
                    "Transformers library docs",
                    "Fine-tune BERT tutorial"
                ]
            elif "fine-tuning" in skill or "lora" in skill:
                resources["fine_tuning"] = [
                    "PEFT library",
                    "LoRA paper",
                    "QLoRA tutorial"
                ]
            elif "rag" in skill:
                resources["rag"] = [
                    "LangChain RAG tutorial",
                    "Build RAG from scratch",
                    "FAISS documentation"
                ]
        return resources

    def generate_recommendation(self, job_text: str) -> LearningRecommendation:
        """Full agent loop: parse → analyze → recommend."""
        start_time = time.time()

        # Step 1: Parse job
        print("Step 1: Parsing job description...")
        job = self.analyze_job(job_text)

        # Step 2: Compute gaps
        print("Step 2: Computing skill gaps...")
        gaps = self.compute_skill_gaps(job)

        # Step 3: Generate path
        print("Step 3: Generating learning path...")
        plan, weeks = self.generate_learning_path(gaps)

        # Step 4: Hardware check
        print("Step 4: Checking hardware compatibility...")
        compatible = find_compatible_models(self.hardware["vram_gb"])
        token_count = estimate_tokens(job_text)
        cost_estimate = calculate_cost(token_count)

        # Build recommendation
        rec = LearningRecommendation(
            summary=f"Your foundation in transformers and VRAM optimization is strong. "
                    f"Focus on {len(gaps)} skills with largest gaps. Estimated {plan.total_estimated_hours} hours over {plan.timeline_weeks} weeks.",
            skill_gaps=gaps,
            learning_path=plan,
            estimated_cost=round(cost_estimate + 25, 2),  # API + recommended courses
            estimated_time=f"{plan.timeline_weeks} weeks, ~15 hours/week",
            hardware_recommendations=[f"Your {self.hardware['gpu']} ({self.hardware['vram_gb']}GB VRAM) supports local models: {', '.join(compatible[:3])}"],
            project_ideas=[
                "Build a VRAM calculator web app (integrate your Day 8 work)",
                "Create a code review agent with structured outputs",
                "Fine-tune a 4B model on your learning notes"
            ]
        )

        elapsed = time.time() - start_time
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "query": job_text[:100],
            "response_time_ms": round(elapsed * 1000, 2),
            "token_count": token_count,
        })

        return rec, weeks, gaps


# --- Demo ---

def demonstrate_agent():
    """Run the full job description analyzer agent."""
    print("=" * 70)
    print("DAY 13: JOB DESCRIPTION ANALYZER AGENT")
    print("The Complete AI Engineering Learning Agent")
    print("=" * 70)

    # Initialize agent
    agent = JobAnalyzerAgent(CURRENT_SKILLS, HARDWARE_PROFILE)

    print(f"\nHardware: {HARDWARE_PROFILE['gpu']} ({HARDWARE_PROFILE['vram_gb']}GB VRAM)")
    print(f"Skills in your toolkit: {len(CURRENT_SKILLS)}")

    # Analyze sample job
    print("\n" + "=" * 70)
    print("ANALYZING: AI Engineer Job Description")
    print("=" * 70)

    recommendation, weeks, gaps = agent.generate_recommendation(SAMPLE_JOB_DESCRIPTION)

    # Display results
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\nSummary: {recommendation.summary}")

    print(f"\nSkill Gaps ({len(gaps)} found):")
    print(f"{'Skill':<30} {'Current':>8} {'Target':>8} {'Gap':>6} {'Hours':>8}")
    print("-" * 65)
    for gap in gaps:
        print(f"{gap['skill']:<30} {gap['current_level']:>8} {gap['target_level']:>8} {gap['gap']:>6} {gap['estimated_hours']:>8}")

    print(f"\nLearning Path ({len(weeks)} weeks):")
    for week in weeks:
        print(f"\n  Week {week['week']}: {week['focus']}")
        print(f"    Hours: {week['hours']}")
        for res in week.get("resources", {}).get("pytorch", [])[:1]:
            print(f"    Resource: {res}")

    print(f"\nHardware Recommendations:")
    for rec in recommendation.hardware_recommendations:
        print(f"  - {rec}")

    print(f"\nProject Ideas:")
    for idea in recommendation.project_ideas:
        print(f"  - {idea}")

    print(f"\nEstimated Cost: ${recommendation.estimated_cost}")
    print(f"Timeline: {recommendation.estimated_time}")

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("1. Your Days 1-12 learning built the right foundation for AI engineering")
    print("2. Largest gaps: PyTorch, Hugging Face, Fine-tuning (need hands-on practice)")
    print("3. Your GTX 1650 can run INT4 quantized 4B models locally")
    print("4. 11-week timeline (~173 hours) to qualify for AI Engineer roles")
    print("5. Project portfolio (agents, VRAM tools) demonstrates practical skills")
    print(f"{'='*70}")


# --- Main Entry ---

if __name__ == "__main__":
    HARDWARE_PROFILE = {
        "gpu": "NVIDIA GTX 1650",
        "vram_gb": 4.0,
        "ram_gb": 16,
        "cpu": "Intel Core i7-10870H",
        "os": "Windows 11"
    }

    demonstrate_agent()
