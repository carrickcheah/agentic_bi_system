import { useState } from 'react';
import { Brain, Database, Zap, Moon, Sun, Settings, User, Bell } from 'lucide-react';

export function Header() {
  const [isDark, setIsDark] = useState(false);
  const [notifications] = useState(3);

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <header className="backdrop-premium border-b border-white/20 dark:border-white/10 px-6 py-4 relative overflow-hidden">
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary-500/5 via-accent-500/5 to-emerald-500/5 animate-gradient-x"></div>
      
      <div className="relative flex items-center justify-between">
        {/* Logo Section */}
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Brain className="w-10 h-10 text-primary-600 dark:text-primary-400 floating-element glow-primary" />
            <div className="absolute inset-0 bg-primary-400 rounded-full blur-xl opacity-20 animate-pulse"></div>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gradient animate-fade-in">
              Agentic BI
            </h1>
            <p className="text-sm text-text-secondary-light dark:text-text-secondary-dark font-medium">
              World-Class Autonomous Intelligence
            </p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center space-x-8">
          {/* Database Status */}
          <div className="glass-card px-4 py-2 interactive-hover">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Database className="w-5 h-5 text-emerald-500" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
              </div>
              <div>
                <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark">
                  Databases
                </p>
                <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark">
                  4 Connected
                </p>
              </div>
            </div>
          </div>

          {/* Cache Performance */}
          <div className="glass-card px-4 py-2 interactive-hover">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Zap className="w-5 h-5 text-accent-500" />
                <div className="absolute inset-0 bg-accent-400 rounded blur-sm opacity-30 animate-glow"></div>
              </div>
              <div>
                <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark">
                  Cache Hit Rate
                </p>
                <div className="flex items-center space-x-2">
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark">
                    68%
                  </p>
                  <div className="w-8 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div className="w-[68%] h-full bg-gradient-to-r from-accent-400 to-primary-400 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-3">
            {/* Notifications */}
            <button className="relative p-2 glass-card interactive-hover rounded-2xl">
              <Bell className="w-5 h-5 text-text-primary-light dark:text-text-primary-dark" />
              {notifications > 0 && (
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-r from-accent-500 to-accent-600 text-white text-xs font-bold rounded-full flex items-center justify-center animate-bounce-gentle">
                  {notifications}
                </div>
              )}
            </button>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 glass-card interactive-hover rounded-2xl relative overflow-hidden group"
            >
              <div className="relative z-10">
                {isDark ? (
                  <Sun className="w-5 h-5 text-amber-500 transition-transform duration-300 group-hover:rotate-90" />
                ) : (
                  <Moon className="w-5 h-5 text-primary-600 transition-transform duration-300 group-hover:-rotate-12" />
                )}
              </div>
              <div className="absolute inset-0 bg-gradient-to-r from-amber-400/20 to-primary-400/20 scale-0 group-hover:scale-100 transition-transform duration-300 rounded-2xl"></div>
            </button>

            {/* Settings */}
            <button className="p-2 glass-card interactive-hover rounded-2xl group">
              <Settings className="w-5 h-5 text-text-primary-light dark:text-text-primary-dark transition-transform duration-300 group-hover:rotate-90" />
            </button>

            {/* User Profile */}
            <button className="glass-card p-2 interactive-hover rounded-2xl status-online">
              <User className="w-5 h-5 text-text-primary-light dark:text-text-primary-dark" />
            </button>
          </div>
        </div>
      </div>

      {/* Ambient particle effect */}
      <div className="absolute top-0 left-1/4 w-2 h-2 bg-primary-400 rounded-full opacity-40 animate-float" style={{ animationDelay: '0s' }}></div>
      <div className="absolute top-4 right-1/3 w-1 h-1 bg-accent-400 rounded-full opacity-60 animate-float" style={{ animationDelay: '2s' }}></div>
      <div className="absolute bottom-2 left-1/2 w-1.5 h-1.5 bg-emerald-400 rounded-full opacity-50 animate-float" style={{ animationDelay: '4s' }}></div>
    </header>
  );
}