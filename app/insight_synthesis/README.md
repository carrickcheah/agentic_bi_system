# Insight Synthesis Module (Phase 5)

## Overview
Phase 5 transforms raw investigation data into strategic business intelligence using AI.

## What It Does
- Takes technical findings from Phase 4
- Uses AI to interpret and explain
- Creates actionable recommendations
- Formats output for different user roles

## Simple Example
**Input**: "Database shows 23% sales drop, 47 complaints, inventory issues"
**Output**: "Sales crisis due to inventory shortage. Fix: Transfer stock from Southwest. Impact: Recover $2.3M/month"

## Components
- `InsightSynthesizer`: Main AI analyst class
- `SynthesisResult`: Structured output with insights and recommendations
- Uses ModelManager from main.py (no separate AI setup)

## Usage
```python
synthesizer = InsightSynthesizer()
result = await synthesizer.synthesize_insights(
    investigation_results=phase_4_results,
    business_context=context,
    user_role="executive",
    output_format=OutputFormat.EXECUTIVE_SUMMARY
)
```