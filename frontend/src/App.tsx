import { useState } from 'react';
import { ConversationPanel } from './components/conversation/ConversationPanel';
import { ResultsPanel } from './components/results/ResultsPanel';
import { Header } from './components/layout/Header';
import { Investigation } from './services/mockApi';

function App() {
  const [currentInvestigation, setCurrentInvestigation] = useState<Investigation | null>(null);
  const [isInvestigating, setIsInvestigating] = useState(false);

  const handleNewQuery = (query: string) => {
    // This will be handled by the ConversationPanel
    setIsInvestigating(true);
  };

  const handleInvestigationUpdate = (investigation: Investigation) => {
    setCurrentInvestigation(investigation);
    if (investigation.status === 'completed' || investigation.status === 'failed') {
      setIsInvestigating(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Conversation */}
        <div className="w-1/2 border-r border-gray-200 flex flex-col">
          <ConversationPanel
            onNewQuery={handleNewQuery}
            onInvestigationUpdate={handleInvestigationUpdate}
            isInvestigating={isInvestigating}
          />
        </div>

        {/* Right Panel - Results */}
        <div className="w-1/2 flex flex-col">
          <ResultsPanel
            investigation={currentInvestigation}
            isInvestigating={isInvestigating}
          />
        </div>
      </div>
    </div>
  );
}

export default App;