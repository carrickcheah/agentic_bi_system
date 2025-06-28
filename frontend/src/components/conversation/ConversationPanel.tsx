import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
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
  resetTrigger?: number;
}

export function ConversationPanel({
  onNewQuery,
  onInvestigationUpdate,
  isInvestigating,
  resetTrigger,
}: ConversationPanelProps) {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: '',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [currentInvestigation, setCurrentInvestigation] = useState<Investigation | null>(null);
  const [showStreamingSQL, setShowStreamingSQL] = useState(false);
  const [streamingCompleted, setStreamingCompleted] = useState(false);
  const [thinkingMode, setThinkingMode] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);
  const [queryQueue, setQueryQueue] = useState<string[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update welcome message when language changes
  useEffect(() => {
    setMessages(prev => prev.map(msg => 
      msg.id === '1' ? { ...msg, content: t('welcome_message') } : msg
    ));
  }, [t]);

  // Reset conversation when resetTrigger changes
  useEffect(() => {
    if (resetTrigger && resetTrigger > 0) {
      console.log('🔧 DEBUG: ConversationPanel resetting due to resetTrigger change');
      
      // Clean up any ongoing investigation
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      
      setMessages([
        {
          id: '1',
          type: 'assistant',
          content: t('welcome_message'),
          timestamp: new Date(),
        },
      ]);
      setCurrentInvestigation(null);
      setShowStreamingSQL(false);
      setStreamingCompleted(false);
    }
  }, [resetTrigger, t]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, []);

  // Process queue when investigation completes
  useEffect(() => {
    if (!isInvestigating && queryQueue.length > 0) {
      // Process the next query in queue
      const nextQuery = queryQueue[0];
      setQueryQueue(prev => prev.slice(1));
      
      // Submit the queued query
      const userMessage: Message = {
        id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'user',
        content: nextQuery,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, userMessage]);
      onNewQuery(nextQuery);
      
      // Continue with the investigation flow
      handleQueuedInvestigation(nextQuery);
    }
  }, [isInvestigating, queryQueue, onNewQuery]);

  const handleQueuedInvestigation = async (query: string) => {
    // Check cache first
    const cacheResult = await mockApi.checkCache(query);
    
    if (cacheResult.hit) {
      // Cache hit - instant response
      const cacheMessage: Message = {
        id: `cache-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'system',
        content: '💾 Cache hit! Retrieved from organizational memory (47ms)',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, cacheMessage]);
    }

    // Start investigation
    const assistantMessage: Message = {
      id: `assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'assistant',
      content: 'Starting investigation...',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);

    // Clean up any previous investigation
    if (cleanupRef.current) {
      cleanupRef.current();
    }

    // Create investigation stream
    cleanupRef.current = createInvestigationStream(query, (investigation) => {
      setCurrentInvestigation(investigation);
      onInvestigationUpdate(investigation);
      
      // Update the assistant message with the investigation
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { 
                ...msg, 
                investigation,
                content: investigation.status === 'completed' 
                  ? `Investigation complete! I found ${investigation.insights.length} key insights. The analysis took ${investigation.executionTime}ms${investigation.cacheHit ? ' (enhanced by cache)' : ''}.`
                  : msg.content
              }
            : msg
        )
      );
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // If investigation is in progress, queue the query
    if (isInvestigating) {
      setQueryQueue(prev => [...prev, input]);
      setInput('');
      
      // Show queued message
      const queuedMessage: Message = {
        id: `queued-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'system',
        content: `🔄 Query queued. Will process after current investigation completes.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, queuedMessage]);
      return;
    }

    const userMessage: Message = {
      id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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
        id: `cache-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'system',
        content: '� Cache hit! Retrieved from organizational memory (47ms)',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, cacheMessage]);
    }

    // Start investigation
    const assistantMessage: Message = {
      id: `assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'assistant',
      content: 'Starting investigation...',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);

    // Clean up any previous investigation
    if (cleanupRef.current) {
      cleanupRef.current();
    }

    // Create investigation stream
    cleanupRef.current = createInvestigationStream(input, (investigation) => {
      setCurrentInvestigation(investigation);
      onInvestigationUpdate(investigation);
      
      // Update the assistant message with the investigation
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { 
                ...msg, 
                investigation,
                content: investigation.status === 'completed' 
                  ? `Investigation complete! I found ${investigation.insights.length} key insights. The analysis took ${investigation.executionTime}ms${investigation.cacheHit ? ' (enhanced by cache)' : ''}.`
                  : 'Starting investigation...'
              } 
            : msg
        )
      );
    });
  };

  const suggestedQueries = [
    t('q4_sales_drop'),
    t('yesterday_sales'),
    t('satisfaction_trends'),
    t('revenue_forecast'),
  ];

  const handleStreamingComplete = () => {
    setStreamingCompleted(true);
    setShowStreamingSQL(true);
  };

  const handleSQLComplete = () => {
    // SQL streaming completed
  };

  return (
    <div className="flex flex-col h-full conversation-panel bg-gray-100 dark:bg-gray-950">
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
                  text={thinkingMode ? `<thinking>
Let me break down this business intelligence query systematically:

1. Query Analysis:
   - Understanding the business context and intent
   - Identifying key metrics, dimensions, and KPIs needed
   - Mapping business language to data warehouse schema

2. Data Source Evaluation:
   - Checking available tables: sales_orders, customers, products, transactions
   - Validating data quality and completeness for the requested timeframe
   - Identifying any potential data gaps or limitations

3. Query Strategy:
   - Determining optimal SQL approach (aggregations vs detailed analysis)
   - Planning JOIN operations for multi-table relationships
   - Considering performance implications for large datasets

4. Business Logic Application:
   - Applying domain-specific calculations (revenue recognition, churn rates, etc.)
   - Incorporating business rules and filters
   - Ensuring compliance with company reporting standards
</thinking>

I'll analyze your business intelligence query step by step.

First, let me understand what you're looking for and identify the key metrics and dimensions needed for this analysis.

Connecting to your business data warehouse... I can see we have access to sales data, customer records, and performance metrics across multiple tables.

Let me examine the data structure and relationships to build the most comprehensive analysis for your query.

Running the analysis now... Processing customer segments, sales trends, and performance indicators.

Based on my analysis of the data, I've identified several key insights that directly address your question. The patterns show some interesting trends that I think you'll find valuable.

Here are the key findings from my investigation:` : `I'll analyze your business intelligence query step by step.

First, let me understand what you're looking for and identify the key metrics and dimensions needed for this analysis.

Connecting to your business data warehouse... I can see we have access to sales data, customer records, and performance metrics across multiple tables.

Let me examine the data structure and relationships to build the most comprehensive analysis for your query.

Running the analysis now... Processing customer segments, sales trends, and performance indicators.

Based on my analysis of the data, I've identified several key insights that directly address your question. The patterns show some interesting trends that I think you'll find valuable.

Here are the key findings from my investigation:`}
                  speed={thinkingMode ? 30 : 50}
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
              {t('try_queries')}
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
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e as any);
                    }
                  }}
                  placeholder={t('ask_placeholder')}
                  className="w-full bg-transparent border-none outline-none text-text-primary-light dark:text-text-primary-dark
                            placeholder:text-text-secondary-light dark:placeholder:text-text-secondary-dark
                            text-sm resize-none"
                  style={{ minHeight: '20px' }}
                />
              </div>
              <button
                type="submit"
                disabled={!input.trim()}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px]"
                style={{ position: 'relative', zIndex: 10 }}
              >
                <div className="flex items-center space-x-2">
                  {isInvestigating ? (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Queue</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>{t('send')}</span>
                    </>
                  )}
                </div>
              </button>
            </div>
          </div>

          {/* Queue indicator */}
          {queryQueue.length > 0 && (
            <div className="mt-2 px-1">
              <div className="text-xs text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/30 px-2 py-1 rounded-full inline-flex items-center space-x-1">
                <span>🔄</span>
                <span>{queryQueue.length} query{queryQueue.length > 1 ? 's' : ''} queued</span>
              </div>
            </div>
          )}

          {/* Input hints */}
          <div className="flex items-center justify-between mt-3 px-1">
            <div className="flex items-center space-x-4 text-xs text-text-secondary-light dark:text-text-secondary-dark">
              {/* Thinking Mode Toggle */}
              <div className="flex items-center space-x-2">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={thinkingMode}
                    onChange={(e) => setThinkingMode(e.target.checked)}
                    className="sr-only"
                  />
                  <div className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${
                    thinkingMode 
                      ? 'bg-gradient-to-r from-blue-500 to-purple-500' 
                      : 'bg-gray-300 dark:bg-gray-600'
                  }`}>
                    <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform duration-200 ${
                      thinkingMode ? 'translate-x-4' : 'translate-x-0'
                    }`}></div>
                  </div>
                  <span className="text-xs">
                    {t('thinking_mode')}
                  </span>
                </label>
              </div>
            </div>
            <div className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
              <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded text-xs">Enter</kbd>
              <span className="ml-1">{t('enter_to_send')}</span>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}