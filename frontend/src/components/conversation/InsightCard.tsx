import { AlertTriangle, Lightbulb, TrendingUp } from 'lucide-react';
import { Insight } from '../../services/mockApi';

interface InsightCardProps {
  insight: Insight;
}

export function InsightCard({ insight }: InsightCardProps) {
  const getIcon = () => {
    switch (insight.type) {
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case 'recommendation':
        return <Lightbulb className="w-5 h-5 text-blue-500" />;
      default:
        return <TrendingUp className="w-5 h-5 text-green-500" />;
    }
  };

  const getBackgroundClass = () => {
    switch (insight.type) {
      case 'warning':
        return 'bg-orange-50 border-orange-200';
      case 'recommendation':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-green-50 border-green-200';
    }
  };

  return (
    <div className={`rounded-lg border p-4 ${getBackgroundClass()}`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-gray-900">{insight.title}</h4>
          <p className="mt-1 text-sm text-gray-600">{insight.description}</p>
          
          {/* Confidence indicator */}
          <div className="mt-2 flex items-center space-x-2">
            <span className="text-xs text-gray-500">Confidence:</span>
            <div className="flex-1 max-w-[100px]">
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div
                  className={`h-1.5 rounded-full ${
                    insight.confidence > 0.8 
                      ? 'bg-green-500' 
                      : insight.confidence > 0.6 
                      ? 'bg-yellow-500' 
                      : 'bg-orange-500'
                  }`}
                  style={{ width: `${insight.confidence * 100}%` }}
                />
              </div>
            </div>
            <span className="text-xs font-medium text-gray-700">
              {Math.round(insight.confidence * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}