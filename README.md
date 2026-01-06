# Multi-Agent Orchestration Spec Language

![License](https://img.shields.io/github/license/ai-engineer-community/multi-agent-orchestrator)
![TypeScript](https://img.shields.io/badge/Made%20with-TypeScript-blue)
![YAML](https://img.shields.io/badge/DSL-YAML-blue)

YAML-based DSL + executor for defining agent roles, tools, dependencies, automatic resource allocation, and built-in error recovery with circuit breakers. Compatible with Anthropic SDK, OpenAI, Groq.

## Features

- YAML-based domain specific language for agent orchestration
- Define agent roles, tools, dependencies
- Automatic resource allocation
- Built-in error recovery & circuit breakers
- Compatible with Anthropic SDK, OpenAI, Groq
- Parallel execution support

## Installation

```bash
npm install multi-agent-orchestrator
```

## Usage

Create a YAML spec file:

```yaml
agents:
  researcher:
    role: "Find research papers"
    tools: [search_arxiv, fetch_pdf]
    max_retries: 3
  analyst:
    role: "Summarize findings"
    inputs: [researcher.output]
    depends_on: [researcher]
```

Run with the executor:

```bash
agent-orchestrate --spec spec.yaml
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT