import { useState, useEffect } from 'react';
import { ConversationPanel } from './components/conversation/ConversationPanel';
import { ResultsPanel } from './components/results/ResultsPanel';
import { Header } from './components/layout/Header';
import { Investigation } from './services/mockApi';

function App() {
  const [currentInvestigation, setCurrentInvestigation] = useState<Investigation | null>(null);
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Simulate app loading
    setTimeout(() => setIsLoaded(true), 800);
  }, []);

  const handleNewQuery = (query: string) => {
    setIsInvestigating(true);
  };

  const handleInvestigationUpdate = (investigation: Investigation) => {
    console.log('🔧 DEBUG: App received investigation update:', investigation);
    setCurrentInvestigation(investigation);
    if (investigation.status === 'completed' || investigation.status === 'failed') {
      console.log('🔧 DEBUG: Setting isInvestigating to false');
      setIsInvestigating(false);
    }
  };

  if (!isLoaded) {
    return (
      <div className="h-screen flex items-center justify-center relative overflow-hidden">
        {/* Animated background */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-accent-50 to-emerald-50 dark:from-primary-900 dark:via-accent-900 dark:to-emerald-900 animate-gradient-y"></div>
        
        {/* Loading animation */}
        <div className="relative z-10 text-center">
          <div className="w-20 h-20 mx-auto mb-8 relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full animate-spin"></div>
            <div className="absolute inset-2 bg-white dark:bg-gray-900 rounded-full flex items-center justify-center">
              <div className="w-8 h-8 bg-gradient-to-r from-primary-400 to-accent-400 rounded-full animate-pulse"></div>
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gradient mb-2">Agentic BI</h1>
          <p className="text-text-secondary-light dark:text-text-secondary-dark">
            Initializing Autonomous Intelligence...
          </p>
        </div>
        
        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-gradient-to-r from-primary-400 to-accent-400 rounded-full opacity-30 animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 6}s`,
                animationDuration: `${6 + Math.random() * 4}s`,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col relative overflow-hidden">
      {/* Dynamic background with mesh gradient */}
      <div className="absolute inset-0 bg-gradient-mesh opacity-5 animate-gradient-x"></div>
      <div className="absolute inset-0 bg-noise opacity-[0.02]"></div>
      
      <Header />
      
      <div className="flex-1 flex overflow-hidden relative z-10">
        {/* Left Panel - Conversation */}
        <div className="w-1/2 flex flex-col relative">
          {/* Gradient divider */}
          <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-primary-200 dark:via-primary-800 to-transparent"></div>
          
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

      {/* Ambient lighting effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-400/10 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-emerald-400/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
    </div>
  );
}

export default App;