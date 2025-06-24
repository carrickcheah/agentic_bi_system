import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../../contexts/ThemeContext';
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
  Settings,
  Moon,
  Sun
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  width: number;
  onNewAnalysis?: () => void;
}

export function Sidebar({ collapsed, onToggle, width, onNewAnalysis }: SidebarProps) {
  const { t, i18n } = useTranslation();
  const { isDarkMode, toggleDarkMode } = useTheme();
  const [selectedChat, setSelectedChat] = useState('agentic-sql-research');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('English (United States)');
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    }

    if (showUserMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showUserMenu]);

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

  const languages = [
    'Bahasa Malaysia',
    '中文 (Chinese)',
    'English (United States)',
    'Français (France)',
    'German (Germany)',
    'हिन्दी (भारत)',
    'Indonesia',
    'Italiano (Italian)',
    'Japanese (Japan)',
    '한국어(대한민국)',
    'Portugal (Brasil)',
    'Español (Latin American)',
    'Español (España)'
  ];

  // Map language display names to i18n language codes
  const languageMap: { [key: string]: string } = {
    'Bahasa Malaysia': 'ms',
    '中文 (Chinese)': 'zh',
    'English (United States)': 'en',
    'Français (France)': 'en', // Fallback to English for now
    'German (Germany)': 'en',
    'हिन्दी (भारत)': 'en',
    'Indonesia': 'en',
    'Italiano (Italian)': 'en',
    'Japanese (Japan)': 'en',
    '한국어(대한민국)': 'en',
    'Portugal (Brasil)': 'en',
    'Español (Latin American)': 'en',
    'Español (España)': 'en'
  };

  const handleLanguageChange = (language: string) => {
    const languageCode = languageMap[language] || 'en';
    i18n.changeLanguage(languageCode);
    setSelectedLanguage(language);
    setShowLanguageMenu(false);
    setShowUserMenu(false);
  };

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
              {t('agentic_bi')}
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
          {!collapsed && <span className="text-sm font-medium">{t('new_analysis')}</span>}
        </button>
      </div>

      {/* Navigation */}
      {!collapsed && (
        <div className="px-3 space-y-1">
          <button className="w-full flex items-center space-x-3 p-2 text-text-secondary-light dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
            <BarChart3 className="w-4 h-4" />
            <span className="text-sm">{t('reports')}</span>
          </button>
          <button className="w-full flex items-center space-x-3 p-2 text-text-secondary-light dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
            <FolderOpen className="w-4 h-4" />
            <span className="text-sm">{t('projects')}</span>
          </button>
        </div>
      )}

      {/* Recent Chats - moved directly under navigation */}
      {!collapsed && (
        <div className="overflow-hidden mt-4">
          <div className="px-3 py-1">
            <h3 className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark uppercase tracking-wider">
              {t('recents')}
            </h3>
          </div>
          <div className="px-2 space-y-0.5 overflow-y-auto max-h-[calc(100vh-300px)]">
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

      {/* Spacer to push user profile to bottom */}
      <div className="flex-1"></div>

      {/* User Profile */}
      <div ref={menuRef} className="relative p-3 border-t border-gray-200 dark:border-gray-700">
        <button 
          onClick={() => {
            if (collapsed) {
              onToggle(); // Expand sidebar when collapsed
            } else {
              setShowUserMenu(!showUserMenu); // Show menu when expanded
            }
          }}
          className="w-full flex items-center space-x-3 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg p-2 -m-2 transition-colors"
        >
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
            C
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <div className="text-left">
                  <p className="text-sm font-medium text-text-primary-light dark:text-text-primary-dark truncate">
                    Carrick
                  </p>
                  <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                    {t('pro_plan')}
                  </p>
                </div>
                <ChevronDown className={`w-4 h-4 text-text-secondary-light dark:text-text-secondary-dark transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
              </div>
            </div>
          )}
        </button>

        {/* User Menu Dropdown */}
        {showUserMenu && !collapsed && (
          <div className="absolute bottom-full left-3 right-3 mb-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg py-2 z-50">
            {/* User Info */}
            <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                  C
                </div>
                <div>
                  <p className="text-sm font-medium text-text-primary-light dark:text-text-primary-dark">
                    carrick113@gmail.com
                  </p>
                  <div className="flex items-center space-x-2 mt-1">
                    <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                      {t('personal')}
                    </p>
                    <div className="w-1 h-1 bg-text-secondary-light dark:bg-text-secondary-dark rounded-full"></div>
                    <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                      {t('pro_plan')}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Menu Items */}
            <div className="py-1">
              <button className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                {t('settings')}
              </button>
              <div className="relative">
                <button 
                  onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                  className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between"
                >
                  <span>{t('language')}</span>
                  <ChevronRight className={`w-4 h-4 transition-transform ${showLanguageMenu ? 'rotate-90' : ''}`} />
                </button>
                
                {/* Language Submenu */}
                {showLanguageMenu && (
                  <div className="absolute left-full top-0 ml-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg py-2 z-50 w-64">
                    {languages.map((language) => (
                      <button
                        key={language}
                        onClick={() => handleLanguageChange(language)}
                        className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between"
                      >
                        <span>{language}</span>
                        {selectedLanguage === language && (
                          <div className="w-4 h-4 flex items-center justify-center">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                {t('get_help')}
              </button>
              <button 
                onClick={toggleDarkMode}
                className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between"
              >
                <span>{t('dark_mode')}</span>
                <div className="flex items-center">
                  {isDarkMode ? (
                    <Moon className="w-4 h-4" />
                  ) : (
                    <Sun className="w-4 h-4" />
                  )}
                </div>
              </button>
              <button className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between">
                <span>{t('learn_more')}</span>
                <ChevronRight className="w-4 h-4" />
              </button>
              <div className="border-t border-gray-200 dark:border-gray-700 mt-1 pt-1">
                <button className="w-full px-4 py-2 text-left text-sm text-text-primary-light dark:text-text-primary-dark hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  {t('log_out')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}