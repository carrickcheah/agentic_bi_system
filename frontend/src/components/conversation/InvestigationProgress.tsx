import { Check, Clock, Loader2, Zap, Target, Search, Layers, Lightbulb, TrendingUp } from 'lucide-react';
import { Investigation } from '../../services/mockApi';

interface InvestigationProgressProps {
  investigation: Investigation;
}

const phaseIcons = {
  'Query Processing': Search,
  'Strategy Planning': Target,
  'Service Orchestration': Layers,
  'Analysis Execution': TrendingUp,
  'Insight Synthesis': Lightbulb,
};

export function InvestigationProgress({ investigation }: InvestigationProgressProps) {
  const completedPhases = investigation.phases.filter(p => p.status === 'completed').length;
  const overallProgress = (completedPhases / investigation.phases.length) * 100;

  return (
    <div className="glass-card p-6 animate-scale-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-2xl flex items-center justify-center">
              <Search className="w-4 h-4 text-white" />
            </div>
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-400 to-accent-400 rounded-2xl blur opacity-30 animate-pulse"></div>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-text-primary-light dark:text-text-primary-dark">
              Investigation Progress
            </h3>
            <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
              {completedPhases} of {investigation.phases.length} phases complete
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <span className={`text-xs px-3 py-1.5 rounded-2xl font-medium ${
            investigation.status === 'completed' 
              ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
              : investigation.status === 'processing'
              ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
              : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
          }`}>
            {investigation.status === 'completed' ? '✨ Complete' : '🔄 In Progress'}
          </span>
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark">
            Overall Progress
          </span>
          <span className="text-xs font-bold text-text-primary-light dark:text-text-primary-dark">
            {Math.round(overallProgress)}%
          </span>
        </div>
        <div className="progress-bar h-3">
          <div 
            className="progress-fill h-full"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Phase Timeline */}
      <div className="space-y-4">
        {investigation.phases.map((phase, index) => {
          const PhaseIcon = phaseIcons[phase.name as keyof typeof phaseIcons] || Search;
          
          return (
            <div key={phase.id} className="relative animate-fade-in" style={{ animationDelay: `${index * 150}ms` }}>
              <div className="flex items-start space-x-4">
                {/* Phase Icon */}
                <div className="flex-shrink-0 relative">
                  <div className={`w-10 h-10 rounded-2xl flex items-center justify-center relative z-10 transition-all duration-300 ${
                    phase.status === 'completed' 
                      ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 shadow-lg shadow-emerald-500/25' 
                      : phase.status === 'in_progress' 
                      ? 'bg-gradient-to-r from-primary-500 to-accent-500 shadow-lg shadow-primary-500/25 animate-glow' 
                      : 'bg-gray-300 dark:bg-gray-600'
                  }`}>
                    {phase.status === 'completed' ? (
                      <Check className="w-5 h-5 text-white" />
                    ) : phase.status === 'in_progress' ? (
                      <PhaseIcon className="w-5 h-5 text-white animate-pulse" />
                    ) : (
                      <PhaseIcon className="w-5 h-5 text-white opacity-60" />
                    )}
                  </div>
                  
                  {/* Glow effect for active phase */}
                  {phase.status === 'in_progress' && (
                    <div className="absolute -inset-1 bg-gradient-to-r from-primary-400 to-accent-400 rounded-2xl blur opacity-40 animate-pulse"></div>
                  )}
                </div>

                {/* Phase Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold text-text-primary-light dark:text-text-primary-dark">
                      {phase.name}
                    </h4>
                    {phase.status === 'in_progress' && (
                      <span className="text-xs text-primary-600 dark:text-primary-400 font-medium">
                        {phase.progress}%
                      </span>
                    )}
                  </div>
                  
                  {/* Individual Phase Progress */}
                  {phase.status === 'in_progress' && (
                    <div className="mb-3">
                      <div className="progress-bar h-2">
                        <div 
                          className="progress-fill h-full"
                          style={{ width: `${phase.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Phase Insights */}
                  {phase.status === 'completed' && phase.insights.length > 0 && (
                    <div className="space-y-1">
                      {phase.insights.map((insight, i) => (
                        <div key={i} className="flex items-start space-x-2 animate-slide-up" style={{ animationDelay: `${i * 100}ms` }}>
                          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                          <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark leading-relaxed">
                            {insight}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Connector Line */}
              {index < investigation.phases.length - 1 && (
                <div className={`absolute left-5 top-12 w-0.5 h-8 transition-colors duration-500 ${
                  phase.status === 'completed' 
                    ? 'bg-gradient-to-b from-emerald-400 to-emerald-300' 
                    : 'bg-gray-300 dark:bg-gray-600'
                }`} />
              )}
            </div>
          );
        })}
      </div>

      {/* Cache Enhancement Badge */}
      {investigation.cacheHit && (
        <div className="mt-6 pt-4 border-t border-white/20 dark:border-white/10">
          <div className="flex items-center justify-between glass-card p-3 rounded-2xl border border-emerald-200/50 dark:border-emerald-800/50">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <Zap className="w-4 h-4 text-emerald-500" />
                <div className="absolute inset-0 bg-emerald-400 rounded blur-sm opacity-30 animate-pulse"></div>
              </div>
              <span className="text-xs font-medium text-emerald-700 dark:text-emerald-300">
                Enhanced by Organizational Cache
              </span>
            </div>
            <div className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold">
              90% Faster
            </div>
          </div>
        </div>
      )}
    </div>
  );
}