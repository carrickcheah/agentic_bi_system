import { Check, Clock, Loader2 } from 'lucide-react';
import { Investigation } from '../../services/mockApi';

interface InvestigationProgressProps {
  investigation: Investigation;
}

export function InvestigationProgress({ investigation }: InvestigationProgressProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-900">Investigation Progress</h3>
        <span className={`text-xs px-2 py-1 rounded-full ${
          investigation.status === 'completed' 
            ? 'bg-green-100 text-green-700'
            : investigation.status === 'processing'
            ? 'bg-blue-100 text-blue-700'
            : 'bg-gray-100 text-gray-700'
        }`}>
          {investigation.status === 'completed' ? 'Complete' : 'In Progress'}
        </span>
      </div>

      <div className="space-y-3">
        {investigation.phases.map((phase, index) => (
          <div key={phase.id} className="relative">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 mt-0.5">
                {phase.status === 'completed' ? (
                  <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                    <Check className="w-3 h-3 text-white" />
                  </div>
                ) : phase.status === 'in_progress' ? (
                  <div className="w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                    <Loader2 className="w-3 h-3 text-white animate-spin" />
                  </div>
                ) : (
                  <div className="w-5 h-5 bg-gray-300 rounded-full flex items-center justify-center">
                    <Clock className="w-3 h-3 text-white" />
                  </div>
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">{phase.name}</p>
                
                {phase.status === 'in_progress' && (
                  <div className="mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${phase.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {phase.status === 'completed' && phase.insights.length > 0 && (
                  <ul className="mt-1 space-y-0.5">
                    {phase.insights.map((insight, i) => (
                      <li key={i} className="text-xs text-gray-600">" {insight}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Connector line */}
            {index < investigation.phases.length - 1 && (
              <div className="absolute left-2.5 top-6 w-0.5 h-full bg-gray-300" />
            )}
          </div>
        ))}
      </div>

      {investigation.cacheHit && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-xs text-green-600 flex items-center">
            <Zap className="w-3 h-3 mr-1" />
            Enhanced by organizational cache
          </p>
        </div>
      )}
    </div>
  );
}