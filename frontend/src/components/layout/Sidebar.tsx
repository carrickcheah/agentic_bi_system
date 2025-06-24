import { useState } from 'react';
import { 
  FolderOpen, 
  Plus, 
  ChevronLeft, 
  ChevronRight,
  User,
  ChevronDown,
  Database,
  BarChart3,
  TrendingUp,
  Users,
  Settings
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  width: number;
  onNewAnalysis?: () => void;
}

export function Sidebar({ collapsed, onToggle, width, onNewAnalysis }: SidebarProps) {
  const [selectedChat, setSelectedChat] = useState('agentic-sql-research');

  const recentChats = [
    'Agentic SQL Agent Research',
    'Q4 Sales Performance Analysis', 
    'Customer Segmentation Study',
    'Revenue Forecast Model',
    'Product Analytics Dashboard',
    'Market Trend Analysis',
    'Operational Efficiency Report',
    'Competitive Analysis Framework',
    'User Behavior Insights',
    'Financial Performance Review',
    'Supply Chain Optimization',
    'Risk Assessment Report'
  ];

  return (
    <div 
      className="flex flex-col bg-surface-light dark:bg-surface-dark border-r border-gray-200 dark:border-gray-700 relative transition-all duration-300"
      style={{ width: collapsed ? '60px' : `${width}px` }}
    >
      {/* Header */}
      <div className={`p-4 border-b border-gray-200 dark:border-gray-700 ${collapsed ? 'flex justify-center' : 'flex items-center justify-between'}`}>
        {!collapsed && (
          <div className="flex items-center space-x-2">
            <Database className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            <span className="font-semibold text-text-primary-light dark:text-text-primary-dark">
              Agentic BI
            </span>
          </div>
        )}
        <button
          onClick={onToggle}
          className={`${collapsed ? 'p-2 rounded-xl' : 'p-2 rounded-xl'} hover:opacity-90 transition-all duration-200`}
          style={{
            background: 'linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%)',
            width: collapsed ? '40px' : 'auto',
            height: collapsed ? '40px' : 'auto'
          }}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4 text-white" />
          ) : (
            <ChevronLeft className="w-4 h-4 text-white" />
          )}
        </button>
      </div>

      {/* New Chat Button */}
      <div className={`p-3 ${collapsed ? 'flex justify-center' : ''}`}>
        <button 
          onClick={onNewAnalysis}
          className={`${collapsed ? 'p-2 rounded-xl' : 'w-full p-3 rounded-lg'} flex items-center ${collapsed ? 'justify-center' : 'space-x-3'} bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white transition-all duration-200 group`}
          style={collapsed ? { width: '40px', height: '40px' } : {}}
        >
          <Plus className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="text-sm font-medium">New Analysis</span>}
        </button>
      </div>

      {/* Navigation */}
      {!collapsed && (
        <div className="px-3 space-y-1">
          <button className="w-full flex items-center space-x-3 p-2 text-text-secondary-light dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
            <BarChart3 className="w-4 h-4" />
            <span className="text-sm">Reports</span>
          </button>
          <button className="w-full flex items-center space-x-3 p-2 text-text-secondary-light dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
            <FolderOpen className="w-4 h-4" />
            <span className="text-sm">Projects</span>
          </button>
        </div>
      )}

      {/* Spacer to push user profile to bottom */}
      <div className="flex-1"></div>

      {/* Recent Chats */}
      {!collapsed && (
        <div className="overflow-hidden">
          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark uppercase tracking-wider">
              Recents
            </h3>
          </div>
          <div className="px-2 space-y-1 overflow-y-auto max-h-[calc(100vh-300px)]">
            {recentChats.map((chat, index) => (
              <button
                key={index}
                onClick={() => setSelectedChat(chat.toLowerCase().replace(/\s+/g, '-'))}
                className={`w-full text-left p-2 rounded-lg transition-colors text-sm truncate ${
                  selectedChat === chat.toLowerCase().replace(/\s+/g, '-')
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'text-text-secondary-light dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title={chat}
              >
                {chat}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* User Profile */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
            C
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-text-primary-light dark:text-text-primary-dark truncate">
                    Carrick
                  </p>
                  <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                    Pro Plan
                  </p>
                </div>
                <ChevronDown className="w-4 h-4 text-text-secondary-light dark:text-text-secondary-dark" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}