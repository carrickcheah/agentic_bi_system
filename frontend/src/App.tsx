import { useState, useEffect } from 'react';
import { ConversationPanel } from './components/conversation/ConversationPanel';
import { ResultsPanel } from './components/results/ResultsPanel';
import { Sidebar } from './components/layout/Sidebar';
import { Investigation } from './services/mockApi';
import './i18n';

function App() {
  const [currentInvestigation, setCurrentInvestigation] = useState<Investigation | null>(null);
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // percentage
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(240); // px

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

  const [resetTrigger, setResetTrigger] = useState(0);

  const handleNewAnalysis = () => {
    // Reset the conversation state
    setCurrentInvestigation(null);
    setIsInvestigating(false);
    
    // Trigger conversation panel reset by incrementing counter
    setResetTrigger(prev => prev + 1);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    const containerWidth = window.innerWidth;
    const newLeftWidth = (e.clientX / containerWidth) * 100;
    
    // Constrain between 20% and 80%
    const constrainedWidth = Math.min(Math.max(newLeftWidth, 20), 80);
    setLeftPanelWidth(constrainedWidth);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging]);

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
    <div className="h-screen flex relative overflow-hidden bg-background-light dark:bg-background-dark">
      {/* Sidebar */}
      <Sidebar 
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        width={sidebarWidth}
        onNewAnalysis={handleNewAnalysis}
      />
      
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
          {/* Left Panel - Conversation */}
          <div 
            className="flex flex-col relative"
            style={{ width: `${leftPanelWidth}%` }}
          >
            <ConversationPanel
              onNewQuery={handleNewQuery}
              onInvestigationUpdate={handleInvestigationUpdate}
              isInvestigating={isInvestigating}
              resetTrigger={resetTrigger}
            />
          </div>

          {/* Resizable Divider */}
          <div 
            className={`relative flex-shrink-0 w-1 cursor-col-resize transition-all duration-200 ${
              isDragging ? 'w-2' : ''
            }`}
            style={{
              background: isDragging 
                ? 'linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%)'
                : 'linear-gradient(to bottom, transparent, rgba(96, 165, 250, 0.3), rgba(167, 139, 250, 0.3), rgba(192, 132, 252, 0.3), transparent)'
            }}
            onMouseDown={handleMouseDown}
          >
            {/* Drag handle indicator */}
            <div 
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-1 h-8 rounded-full opacity-0 hover:opacity-100 transition-opacity duration-200"
              style={{
                background: 'linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%)'
              }}
            ></div>
          </div>

          {/* Right Panel - Results */}
          <div 
            className="flex flex-col"
            style={{ width: `${100 - leftPanelWidth}%` }}
          >
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