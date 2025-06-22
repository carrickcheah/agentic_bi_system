# AI Model Integration Guide

## Overview

This directory contains AI model integrations for the Agentic SQL Backend, implementing a cost-optimized fallback system with three tiers: Anthropic Claude Sonnet 4 (primary), DeepSeek (secondary), and OpenAI GPT-4.1 nano (tertiary).

## Model Specifications

| Model | Version | Context Window | Strengths | Limitations |
|-------|---------|----------------|-----------|-------------|
| Claude Sonnet 4 | claude-sonnet-4-20250514 | 200K | Advanced reasoning, prompt caching | Higher cost |
| DeepSeek Reasoner | deepseek-reasoner | 64K | Deep-thinking, cost-effective | Higher latency for reasoning |
| DeepSeek Chat | deepseek-chat | 64K | Ultra-low cost, fast | Basic reasoning only |
| OpenAI GPT-4.1 nano* | gpt-4.1-nano | 1M | Fastest, cheapest OpenAI model | Limited availability |

*GPT-4.1 nano scores 80.1% on MMLU, ideal for classification/autocompletion

## Pricing Comparison

### Base Pricing (per million tokens)

| Model | Input (Standard) | Input (Discount) | Output (Standard) | Output (Discount) | Cache Hit | Notes |
|-------|------------------|------------------|-------------------|-------------------|-----------|-------|
| Claude Sonnet 4 | $3.00 | N/A | $15.00 | N/A | $0.30 | Cache: 5min($3.75), 1hr($6.00) |
| DeepSeek Reasoner | $0.55 | $0.14 (75% off) | $2.19 | $0.55 (75% off) | $0.14 | Deep-thinking model |
| DeepSeek Chat | $0.27 | $0.14 (50% off) | $1.10 | $0.55 (50% off) | $0.07 | Fast model |
| OpenAI GPT-4.1 nano** | $0.10* | N/A | $0.40* | N/A | N/A | 1M context, estimated pricing |

*Estimated pricing for GPT-4.1 nano (awaiting official announcement)
**DeepSeek discount periods: UTC 16:30-00:30 (Asian timezone optimization)

### Cost Analysis: Typical SQL Investigation
**Scenario**: 10K system prompt + 2K query + 3K response

| Configuration | Standard Hours | Discount Hours | vs DeepSeek Chat | Best Use Case |
|---------------|----------------|----------------|------------------|---------------|
| Claude (5min cache) | $0.0885 | N/A | 42.1x | Active sessions |
| Claude (1hr cache) | $0.111 | N/A | 52.9x | Distributed usage |
| Claude (Batch) | $0.0405 | N/A | 19.3x | Bulk processing |
| DeepSeek Reasoner | $0.0123 | $0.0035 | 5.9x / 1.7x | Complex reasoning |
| DeepSeek Chat | $0.0057 | $0.0021 | 2.7x / 1.0x | High volume |
| GPT-4.1 nano* | $0.0024 | N/A | 1.1x | Fast classification |

*Based on estimated pricing
**DeepSeek discount: UTC 16:30-00:30 (75% off reasoner, 50% off chat)

## Annual Cost Projections

### By Volume Tier

| Volume | Queries/Year | Daily Average | Claude Only | Smart Routing* | DeepSeek Reasoner** | DeepSeek Chat | GPT-4.1 nano |
|--------|--------------|---------------|-------------|----------------|-------------------|---------------|---------------|
| Low | 100K | 274 | $5,450 | $1,230 | $350 | $210 | $240 |
| Medium | 500K | 1,370 | $27,250 | $6,150 | $1,750 | $1,050 | $1,200 |
| High | 1M | 2,740 | $54,500 | $12,300 | $3,500 | $2,100 | $2,400 |
| Enterprise | 2M | 5,480 | $109,000 | $24,600 | $7,000 | $4,200 | $4,800 |

*Smart Routing: 70% DeepSeek Reasoner, 30% Claude
**DeepSeek Reasoner pricing assumes 50% during discount hours

### Cost Breakdown by Use Case

| Use Case | Daily Queries | Annual Queries | Claude Cost | Smart Route | DeepSeek | Savings |
|----------|---------------|----------------|-------------|-------------|----------|---------|
| 10 Analysts | 500 | 182,500 | $9,955 | $3,249 | $383 | 67% |
| BI Dashboard | 2,000 | 730,000 | $39,785 | $12,994 | $1,533 | 67% |
| Customer Analytics | 2,500 | 912,500 | $49,731 | $16,243 | $1,916 | 67% |

## Implementation Architecture

### Fallback Chain

| Priority | Model | Trigger | Purpose |
|----------|-------|---------|---------|
| 1 | Claude Sonnet 4 | Default | Complex reasoning, best accuracy |
| 2 | DeepSeek | Claude fails | Ultra-low cost alternative |
| 3 | GPT-4.1 nano | Both fail | Fast, reliable, cost-effective |

### Alternative Strategy for High-Volume Simple Queries

| Priority | Model | Trigger | Purpose |
|----------|-------|---------|---------|
| 1 | DeepSeek | Default | Cheapest option |
| 2 | GPT-4.1 nano | DeepSeek fails | Fast, slightly higher cost |
| 3 | Claude Sonnet 4 | Both fail | Premium accuracy |

### Model Selection Criteria

| Query Type | Recommended Model | Reasoning |
|------------|-------------------|-----------|
| Complex SQL Analysis | Claude | Advanced reasoning required |
| Schema Exploration | Claude (cached) | Benefits from context caching |
| Simple Queries | DeepSeek | Best cost optimization |
| Classification/Validation | GPT-4.1 nano | Fast, optimized for classification |
| Bulk Operations | DeepSeek or GPT-4.1 nano Batch | Volume discounts |
| Critical Queries | Claude | Accuracy paramount |
| Real-time queries | GPT-4.1 nano | Lowest latency |

### Caching Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| anthropic_enable_caching | true | Enable prompt caching |
| cache_system_prompt | true | Cache system prompts |
| cache_schema_info | true | Cache database schemas |
| prompt_cache_ttl | 3600 | Cache TTL in seconds |

## Performance Metrics

### Response Times (Average)

| Model | First Request | Cached Request | Batch |
|-------|---------------|----------------|-------|
| Claude | 2.4s | 0.8s | 1-24hr |
| DeepSeek | 1.2s | N/A | N/A |
| OpenAI | 1.0s | N/A | N/A |

### Success Rates

| Model | Availability | Error Rate | Fallback Trigger |
|-------|--------------|------------|------------------|
| Claude | 99.9% | <0.1% | Rate limits |
| DeepSeek | 99.5% | <0.5% | Capacity |
| OpenAI | 99.9% | <0.1% | API errors |

## Cost Optimization Strategies

### By Query Volume

| Volume Range | Strategy | Expected Savings |
|--------------|----------|------------------|
| <100K/year | Claude only | Baseline |
| 100K-500K | Smart routing | 50-60% |
| 500K-1M | 70/30 routing | 65-70% |
| >1M | 80/20 routing | 70-75% |

### By Query Complexity

| Complexity | Routing Decision | Cost Impact |
|------------|------------------|-------------|
| Simple SELECT | DeepSeek | -96% |
| JOIN queries | DeepSeek/Claude | -60% |
| Complex analytics | Claude | Baseline |
| Multi-step | Claude (cached) | -39% |

## Claude Sonnet 4 Caching Strategy

### Cache Type Comparison

| Cache Type | Write Cost | Read Cost | Break-even | Best For |
|------------|------------|-----------|------------|----------|
| 5-minute | +25% | -90% | 2 requests | Active sessions |
| 1-hour | +100% | -90% | 3 requests | Distributed sessions |
| Batch | -50% | N/A | Immediate | Bulk processing |

### Usage Pattern Recommendations

| Usage Pattern | Cache Strategy | Expected Savings |
|---------------|----------------|------------------|
| Rapid queries (<5min apart) | 5-minute cache | 39% per session |
| Sporadic queries | 1-hour cache | 35% per session |
| Bulk analysis | Batch API | 54% always |
| Mixed patterns | 5-min default | 30-40% average |


## Maintenance

### Monitoring Checklist

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API Success Rate | >99% | <95% |
| Response Time | <3s | >5s |
| Cost per Query | <$0.02 | >$0.05 |
| Cache Hit Rate | >80% | <60% |

### Regular Tasks

| Task | Frequency | Purpose |
|------|-----------|---------|
| Check API limits | Daily | Prevent failures |
| Review costs | Weekly | Optimization |
| Update API keys | Monthly | Security |
| Test fallbacks | Monthly | Reliability |

## Query Volume Calculator

### Daily Query Estimates by Team Size

| Team Size | Queries/Person/Day | Daily Total | Annual Total | Recommended Strategy |
|-----------|-------------------|-------------|--------------|---------------------|
| 5 analysts | 50 | 250 | 91,250 | Claude primary |
| 10 analysts | 50 | 500 | 182,500 | Smart routing |
| 20 analysts | 50 | 1,000 | 365,000 | 70/30 routing |
| 50 analysts | 50 | 2,500 | 912,500 | 80/20 routing |

### Automated System Projections

| System Type | Reports/Day | Queries/Report | Daily Total | Annual Total |
|-------------|-------------|----------------|-------------|--------------|
| Small BI | 50 | 5 | 250 | 91,250 |
| Medium BI | 200 | 10 | 2,000 | 730,000 |
| Enterprise BI | 500 | 15 | 7,500 | 2,737,500 |
| Real-time Analytics | 1,000 | 20 | 20,000 | 7,300,000 |

## Key Insights with GPT-4.1 nano

### Cost Efficiency Comparison (per investigation)

| Model | Standard Cost | Discount Cost | Speed | Reasoning | Best For |
|-------|---------------|---------------|-------|-----------|----------|
| DeepSeek Chat | $0.0057 | $0.0021 | Fast | Basic | High volume, cost-critical |
| GPT-4.1 nano | $0.0024* | N/A | Fastest | Good | Real-time, classification |
| DeepSeek Reasoner | $0.0123 | $0.0035 | Medium | Deep | Complex SQL analysis |
| Claude Sonnet 4 | $0.054 | N/A | Medium | Excellent | Advanced reasoning |

*Estimated pricing

### Updated Recommendations

1. **Premium Strategy**: Claude → DeepSeek Reasoner → GPT-4.1 nano
2. **Balanced Strategy**: DeepSeek Reasoner → Claude → GPT-4.1 nano  
3. **Cost-Optimized Strategy**: DeepSeek Chat → DeepSeek Reasoner → Claude
4. **Real-Time Strategy**: GPT-4.1 nano → DeepSeek Chat → Claude

### DeepSeek Reasoner Benefits

- **Deep-thinking capabilities** with step-by-step reasoning
- **75% discount during Asian hours** (UTC 16:30-00:30)
- **Cost-effective reasoning** compared to Claude
- **Better for complex SQL analysis** than basic chat models

### Timezone Optimization

| Hours (UTC) | DeepSeek Pricing | Recommended Usage |
|-------------|------------------|-------------------|
| 00:30-16:30 | Standard | Light usage, critical queries only |
| 16:30-00:30 | 75% discount | Heavy processing, bulk analysis |

## Future Optimizations

| Optimization | Potential Savings | Implementation Effort |
|--------------|-------------------|----------------------|
| Query complexity routing | 20-30% | Medium |
| GPT-4.1 nano for classification | 10-15% | Low |
| Request batching | 15-25% | Low |
| Enhanced caching strategy | 10-15% | Low |
| Multi-region deployment | 5-10% | High |

---

*Last Updated: January 2025 - Now includes GPT-4.1 nano integration*