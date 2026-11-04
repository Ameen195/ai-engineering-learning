# Day 13: Job Description Analyzer Agent

## What We Built

A **capstone AI agent** that analyzes job descriptions and generates personalized learning roadmaps, integrating all concepts from Days 1-12.

## Architecture

```
Job Description
      ↓
[Agent Loop]
├── Step 1: Parse → Structured JSON (skills, experience)
├── Step 2: Compute gaps → Compare against your knowledge
├── Step 3: Generate path → Weekly learning plan
├── Step 4: Check hardware → Which models you can run
└── Return: LearningRecommendation
```

## Key Components

### 1. Structured Output for Job Parsing

```python
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
                    "priority": {"type": "string", "enum": ["required", "preferred"]}
                }
            }
        }
    }
}
```

The agent extracts structured data from free text — title, skills, priorities, experience requirements.

### 2. Skill Gap Analysis

```python
def compute_skill_gaps(job_skills, my_skills):
    gaps = []
    for skill in job_skills:
        if skill in my_skills:
            gap = target_level - current_level
        else:
            gap = 2  # New skill = 2 levels
        gaps.append({"skill": skill, "gap": gap, "hours": gap * 10})
    return gaps
```

Compares job requirements against your current skill list (based on Days 1-12 learning).

### 3. Optimized Learning Path Generation

- Sort gaps by priority
- Pack into weekly blocks (~15 hours/week)
- Assign relevant resources for each week
- Track total estimated hours

## Integration with Previous Days

| Day | Component | Integration |
|-----|-----------|-------------|
| 1 | Token counting | Estimate API costs |
| 3 | Cost tracking | Estimate learning resource costs |
| 4 | Model comparison | Find compatible local models |
| 5 | Code review | Assess skill level patterns |
| 6-7 | Transformers | Core knowledge for gap analysis |
| 8 | VRAM calculation | Hardware recommendations |
| 10-11 | Structured outputs | Job parsing + learning path |
| 12 | Provider abstraction | Future API integration |

## Sample Output

For an "AI Engineer" job posting, the agent found:
- **7 skill gaps** (PyTorch, Hugging Face, Fine-tuning, RAG, etc.)
- **Estimated 52 hours** over 4 weeks
- **Compatible models** for your GTX 1650
- **Project ideas** to build your portfolio

## Files Created

- `experiments/13_job_analyzer_agent.py` - Complete capstone agent
- `notes/day13_job_analyzer.md` - This file

## Run

```bash
python experiments/13_job_analyzer_agent.py
```

## Next Steps

1. Add real LLM integration (from Day 12 provider abstraction)
2. Accept user's actual job descriptions as input
3. Allow custom skill levels input
4. Add progress tracking across sessions
