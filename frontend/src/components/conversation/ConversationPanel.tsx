import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Brain, Zap } from 'lucide-react';
import { Investigation, createInvestigationStream, mockApi } from '../../services/mockApi';
import { InvestigationProgress } from './InvestigationProgress';
import { InsightCard } from './InsightCard';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  investigation?: Investigation;
}

interface ConversationPanelProps {
  onNewQuery: (query: string) => void;
  onInvestigationUpdate: (investigation: Investigation) => void;
  isInvestigating: boolean;
}

export function ConversationPanel({
  onNewQuery,
  onInvestigationUpdate,
  isInvestigating,
}: ConversationPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your autonomous business intelligence analyst. Ask me anything about your business data - from simple metrics to complex investigations.',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [currentInvestigation, setCurrentInvestigation] = useState<Investigation | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isInvestigating) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    onNewQuery(input);

    // Check cache first
    const cacheResult = await mockApi.checkCache(input);
    
    if (cacheResult.hit) {
      // Cache hit - instant response
      const cacheMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '¡ Cache hit! Retrieved from organizational memory (47ms)',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, cacheMessage]);
    }

    // Start investigation
    const assistantMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: 'Starting investigation...',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);

    // Create investigation stream
    const cleanup = createInvestigationStream(input, (investigation) => {
      setCurrentInvestigation(investigation);
      onInvestigationUpdate(investigation);
      
      // Update the assistant message with the investigation
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { ...msg, investigation } 
            : msg
        )
      );

      // Add completion message
      if (investigation.status === 'completed') {
        const completionMessage: Message = {
          id: Date.now().toString(),
          type: 'assistant',
          content: `Investigation complete! I found ${investigation.insights.length} key insights. The analysis took ${investigation.executionTime}ms${investigation.cacheHit ? ' (enhanced by cache)' : ''}.`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, completionMessage]);
      }
    });

    // Cleanup function will be called when component unmounts
    return cleanup;
  };

  const suggestedQueries = [
    "Why did Q4 sales drop?",
    "What were yesterday's sales?",
    "Customer satisfaction trends",
    "Revenue forecast next quarter",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : ''}`}>
              {message.type === 'assistant' && (
                <div className="flex items-center space-x-2 mb-2">
                  <Brain className="w-5 h-5 text-primary-600" />
                  <span className="text-sm font-medium text-gray-700">Business Analyst</span>
                </div>
              )}
              {message.type === 'system' && (
                <div className="flex items-center space-x-2 mb-2">
                  <Zap className="w-5 h-5 text-green-500" />
                  <span className="text-sm font-medium text-green-700">System</span>
                </div>
              )}
              
              <div className={`rounded-lg px-4 py-3 ${
                message.type === 'user' 
                  ? 'bg-primary-600 text-white' 
                  : message.type === 'system'
                  ? 'bg-green-50 text-green-800 border border-green-200'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              </div>

              {message.investigation && (
                <div className="mt-4 space-y-4">
                  <InvestigationProgress investigation={message.investigation} />
                  
                  {message.investigation.status === 'completed' && (
                    <div className="space-y-3">
                      {message.investigation.insights.map((insight) => (
                        <InsightCard key={insight.id} insight={insight} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested queries */}
      {messages.length === 1 && (
        <div className="px-6 pb-4">
          <p className="text-sm text-gray-500 mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQueries.map((query) => (
              <button
                key={query}
                onClick={() => setInput(query)}
                className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-gray-200 p-4">
        <div className="flex space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your business data..."
            disabled={isInvestigating}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || isInvestigating}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {isInvestigating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Investigating...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Send</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}