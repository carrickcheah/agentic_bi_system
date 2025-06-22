# Agentic SQL Web UI

> Claude.ai-style interface for autonomous SQL investigation

## Overview

This is the web-based user interface for the Agentic SQL autonomous data investigation agent. Built with React and TypeScript, it provides a clean, intuitive interface for interacting with the SQL agent while watching it work through complex data analysis tasks in real-time.

## UI Design

The interface follows Claude.ai's elegant two-panel design:

```
┌─────────────────┬─────────────────┐
│                 │                 │
│   Conversation  │   Live Results  │
│      Panel      │     Panel       │
│                 │                 │
│  - User queries │ - SQL queries   │
│  - Agent        │ - Data tables   │
│    responses    │ - Charts/graphs │
│  - Progress     │ - Error logs    │
│    updates      │ - Execution     │
│                 │   metrics       │
└─────────────────┴─────────────────┘
```

## Features

### Conversation Panel (Left)
- **Natural Language Input**: Ask business questions in plain English
- **Agent Responses**: See the agent's analysis and insights
- **Progress Tracking**: Real-time updates on investigation steps
- **Investigation History**: Scroll through past queries and results

### Live Results Panel (Right)
- **SQL Query Display**: View generated queries with syntax highlighting
- **Data Tables**: Interactive tables with sorting and filtering
- **Visualizations**: Auto-generated charts based on data type
- **Error Logs**: Detailed error messages and debugging info
- **Performance Metrics**: Query execution times and optimization suggestions

## Technology Stack

- **Framework**: React 18+ with TypeScript
- **UI Components**: Shadcn/ui + Tailwind CSS
- **State Management**: Zustand
- **Charts**: Recharts
- **Tables**: TanStack Table
- **Code Editor**: Monaco Editor (for SQL display)
- **Real-time Communication**: WebSockets
- **Build Tool**: Vite

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Running Agentic SQL backend server

### Installation

1. Navigate to the web app directory:
```bash
cd ui/web_app
```

2. Install dependencies:
```bash
npm install
```

3. Copy environment variables:
```bash
cp .env.example .env
```

4. Configure the backend API endpoint in `.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

## Project Structure

```
web_app/
├── src/
│   ├── components/
│   │   ├── ConversationPanel/
│   │   │   ├── MessageList.tsx
│   │   │   ├── InputBox.tsx
│   │   │   └── ProgressIndicator.tsx
│   │   ├── ResultsPanel/
│   │   │   ├── SQLViewer.tsx
│   │   │   ├── DataTable.tsx
│   │   │   ├── ChartContainer.tsx
│   │   │   └── MetricsDisplay.tsx
│   │   └── shared/
│   │       ├── Layout.tsx
│   │       └── ThemeToggle.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   └── useAgentState.ts
│   ├── stores/
│   │   └── conversationStore.ts
│   ├── types/
│   │   └── index.ts
│   └── utils/
│       ├── chartHelpers.ts
│       └── dataFormatters.ts
├── public/
├── index.html
└── package.json
```

## Key Components

### ConversationPanel
Handles user input and displays the conversation flow with the agent.

```typescript
interface Message {
  id: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  investigationSteps?: InvestigationStep[];
}
```

### ResultsPanel
Displays query results, visualizations, and performance metrics.

```typescript
interface QueryResult {
  sql: string;
  data: any[];
  executionTime: number;
  rowCount: number;
  visualization?: ChartConfig;
}
```

### WebSocket Integration
Real-time updates from the agent during investigation.

```typescript
interface AgentUpdate {
  type: 'progress' | 'result' | 'error' | 'complete';
  data: any;
  timestamp: Date;
}
```

## Styling

The UI uses Tailwind CSS with a custom theme that matches Claude.ai's aesthetic:
- Clean, minimal design
- Subtle animations
- Professional color palette
- Responsive layout

## Contributing

1. Follow the existing code style
2. Write tests for new features
3. Update documentation as needed
4. Submit PR with clear description

## License

Part of the Agentic SQL project - see main repository for license details.