# LLM System Prompt Outline

## 1. Core Identity & Role Definition

- **Primary Role**: Define the AI's main function and expertise area
- **Use Case Specification**: Clarify intended applications and target audience
- **Personality & Tone**: Establish communication style (professional, helpful, creative, etc.)
- **User Relationship**: Set expectations for interaction dynamics

## 2. Behavioral Guidelines

- **Communication Style**: Tone, formality level, and language preferences
- **Response Format**: Structure requirements and presentation standards
- **Detail Level**: When to provide brief vs. comprehensive responses
- **Clarification Protocol**: Guidelines for asking questions vs. making assumptions

## 3. Knowledge Boundaries & Limitations

- **Knowledge Cutoff**: How to handle information beyond training data
- **Topic Restrictions**: Areas to avoid or redirect appropriately
- **Uncertainty Handling**: When and how to acknowledge knowledge gaps
- **Scope Management**: Responses to requests outside expertise area

## 4. Output Formatting & Structure

- **Format Preferences**: Markdown, code blocks, lists, tables, etc.
- **Citation Standards**: Source attribution and reference requirements
- **Code Documentation**: Standards for technical content presentation
- **Error Communication**: How to present troubleshooting and solutions

## 5. Safety & Ethical Guidelines

- **Content Filtering**: Rules for inappropriate or harmful content
- **Bias Mitigation**: Instructions for fair and inclusive responses
- **Privacy Protection**: Data handling and confidentiality protocols
- **Harm Prevention**: Detection and response to potentially dangerous requests

## 6. Context Management

- **Conversation History**: How to utilize previous exchanges effectively
- **Memory Management**: State preservation and information retention
- **Multi-turn Handling**: Maintaining coherence across extended conversations
- **Context Optimization**: Efficient use of available context window

## 7. Tool Usage & Integration

- **Tool Selection**: When and which tools to use for different tasks
- **API Interactions**: Protocols for external service integration
- **File Processing**: Guidelines for handling uploads and data analysis
- **Error Recovery**: Fallback strategies when tools fail

## 8. Domain-Specific Instructions

- **Technical Accuracy**: Standards for precision and correctness
- **Industry Standards**: Adherence to best practices and conventions
- **Terminology Usage**: Appropriate technical language and jargon
- **Validation Processes**: Methods for verifying information and solutions

## 9. Edge Case Handling

- **Ambiguity Resolution**: Strategies for unclear or conflicting requests
- **Priority Management**: How to handle competing instructions
- **Fallback Behaviors**: Default responses when standard approaches fail
- **Escalation Procedures**: When to redirect or seek additional input

## 10. Testing & Validation Framework

- **Success Metrics**: Criteria for evaluating response quality
- **Failure Prevention**: Common pitfalls and how to avoid them
- **Performance Standards**: Benchmarks for response time and accuracy
- **Improvement Protocol**: Mechanisms for ongoing refinement and updates

---

## Implementation Notes

### Structure Template
```
# System Prompt: [AI Name/Role]

## Core Instructions
[Primary behavioral guidelines]

## Response Format
[Output specifications]

## Knowledge & Limitations
[Boundary definitions]

## Safety & Ethics
[Protection protocols]

## Examples
[Demonstration cases]
```

### Best Practices
- Keep instructions clear and unambiguous
- Prioritize safety and ethical considerations
- Include concrete examples for complex behaviors
- Test with edge cases and adversarial inputs
- Regularly update based on performance feedback