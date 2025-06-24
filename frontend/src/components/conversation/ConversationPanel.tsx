import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Brain, Zap, Sparkles, MessageCircle, ArrowUp } from 'lucide-react';
import { Investigation, createInvestigationStream, mockApi } from '../../services/mockApi';
import { InvestigationProgress } from './InvestigationProgress';
import { InsightCard } from './InsightCard';
import { StreamingResults } from '../results/StreamingResults';
import { StreamingSQL } from '../results/StreamingSQL';
import { StreamingText } from '../common/StreamingText';

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
  const [showStreamingSQL, setShowStreamingSQL] = useState(false);
  const [streamingCompleted, setStreamingCompleted] = useState(false);

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
        content: '� Cache hit! Retrieved from organizational memory (47ms)',
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

  const handleStreamingComplete = () => {
    setStreamingCompleted(true);
    setShowStreamingSQL(true);
    
    // Create investigation result for right panel
    const investigation: Investigation = {
      id: Date.now().toString(),
      query: input,
      status: 'completed',
      complexity: 'analytical',
      phases: [
        {
          id: '1',
          name: 'Query Analysis',
          status: 'completed',
          progress: 100,
          insights: ['Identified key metrics and dimensions']
        },
        {
          id: '2', 
          name: 'Data Processing',
          status: 'completed',
          progress: 100,
          insights: ['Processed 247,891 records', 'Found significant trends']
        }
      ],
      insights: [
        {
          id: '1',
          type: 'finding',
          title: 'Sales Performance Trend',
          description: 'Q4 sales increased by 23% compared to previous quarter, driven by strong performance in enterprise segment.',
          confidence: 0.94,
          data: {
            metric: 'sales_growth',
            value: 0.23,
            period: 'Q4'
          }
        },
        {
          id: '2',
          type: 'recommendation', 
          title: 'Focus on Enterprise Segment',
          description: 'Consider increasing investment in enterprise sales team based on strong Q4 performance.',
          confidence: 0.87
        }
      ],
      createdAt: new Date(),
      completedAt: new Date(),
      cacheHit: false,
      executionTime: 1250
    };
    
    console.log('🔧 DEBUG: Streaming completed, updating investigation:', investigation);
    
    // Trigger the investigation update to show results in right panel
    onInvestigationUpdate(investigation);
    setCurrentInvestigation(investigation);
  };

  const handleSQLComplete = () => {
    // SQL streaming completed
  };

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
              
              <div className={`relative ${
                message.type === 'user' 
                  ? 'message-user' 
                  : message.type === 'system'
                  ? 'message-system'
                  : 'message-assistant'
              }`}>
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                
                {/* Message timestamp */}
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-white/10">
                  <span className="text-xs opacity-70">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                  {message.type === 'assistant' && (
                    <div className="flex items-center space-x-1">
                      <Sparkles className="w-3 h-3 opacity-60" />
                      <span className="text-xs opacity-70">AI Generated</span>
                    </div>
                  )}
                </div>
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
        
        {/* Claude Code Style Streaming */}
        {isInvestigating && (
          <div className="flex justify-start">
            <div className="max-w-[80%]">
              <div className="flex items-center space-x-2 mb-2">
                <Brain className="w-5 h-5 text-primary-600" />
                <span className="text-sm font-medium text-gray-700">Assistant</span>
              </div>
              
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl p-4 shadow-sm">
                <StreamingText 
                  text={`I'll analyze your business intelligence query step by step.

First, let me understand what you're looking for and identify the key metrics and dimensions needed for this analysis.

Connecting to your business data warehouse... I can see we have access to sales data, customer records, and performance metrics across multiple tables.

Let me examine the data structure and relationships to build the most comprehensive analysis for your query.

Running the analysis now... Processing customer segments, sales trends, and performance indicators.

Based on my analysis of the data, I've identified several key insights that directly address your question. The patterns show some interesting trends that I think you'll find valuable.

Here are the key findings from my investigation:`}
                  speed={50}
                  showCursor={true}
                  onComplete={handleStreamingComplete}
                  className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-line leading-relaxed"
                />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested queries */}
      {messages.length === 1 && (
        <div className="px-6 pb-6 animate-fade-in">
          <div className="flex items-center space-x-2 mb-4">
            <MessageCircle className="w-4 h-4 text-primary-500" />
            <p className="text-sm font-medium text-text-secondary-light dark:text-text-secondary-dark">
              Try these powerful queries:
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {suggestedQueries.map((query, index) => (
              <button
                key={query}
                onClick={() => setInput(query)}
                className="glass-card px-4 py-3 text-left text-sm text-text-primary-light dark:text-text-primary-dark
                          hover:scale-[1.02] transition-all duration-300 group
                          border border-primary-200/50 dark:border-primary-800/50"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{query}</span>
                  <ArrowUp className="w-3 h-3 opacity-0 group-hover:opacity-60 transition-opacity rotate-45" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Section */}
      <div className="backdrop-premium border-t border-white/20 dark:border-white/10 p-6">
        <form onSubmit={handleSubmit} className="relative">
          <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-xl p-4 rounded-3xl border border-primary-200/50 dark:border-primary-800/50 
                          focus-within:border-primary-400 dark:focus-within:border-primary-600 
                          transition-all duration-300">
            <div className="flex items-end space-x-4">
              <div className="flex-1">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask me anything about your business intelligence..."
                  disabled={isInvestigating}
                  className="w-full bg-transparent border-none outline-none text-text-primary-light dark:text-text-primary-dark
                            placeholder:text-text-secondary-light dark:placeholder:text-text-secondary-dark
                            text-sm resize-none disabled:opacity-50"
                  style={{ minHeight: '20px' }}
                />
              </div>
              <button
                type="submit"
                disabled={!input.trim() || isInvestigating}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px]"
                style={{ position: 'relative', zIndex: 10 }}
              >
                <div className="flex items-center space-x-2">
                  {isInvestigating ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Investigating</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Send</span>
                    </>
                  )}
                </div>
              </button>
            </div>
          </div>

          {/* Input hints */}
          <div className="flex items-center justify-between mt-3 px-1">
            <div className="flex items-center space-x-4 text-xs text-text-secondary-light dark:text-text-secondary-dark">
              <span className="flex items-center space-x-1">
                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span>
                <span>Powered by Claude Sonnet 4.0</span>
              </span>
              <span className="flex items-center space-x-1">
                <Zap className="w-3 h-3" />
                <span>Intelligent caching enabled</span>
              </span>
            </div>
            <div className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
              <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded text-xs">Enter</kbd>
              <span className="ml-1">to send</span>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}