import { Brain } from 'lucide-react';

export function Header() {

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

      </div>

      {/* Ambient particle effect */}
      <div className="absolute top-0 left-1/4 w-2 h-2 bg-primary-400 rounded-full opacity-40 animate-float" style={{ animationDelay: '0s' }}></div>
      <div className="absolute top-4 right-1/3 w-1 h-1 bg-accent-400 rounded-full opacity-60 animate-float" style={{ animationDelay: '2s' }}></div>
      <div className="absolute bottom-2 left-1/2 w-1.5 h-1.5 bg-emerald-400 rounded-full opacity-50 animate-float" style={{ animationDelay: '4s' }}></div>
    </header>
  );
}