import { Brain, Database, Zap } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Agentic BI</h1>
            <p className="text-sm text-gray-500">Autonomous Business Intelligence</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2 text-sm">
            <Database className="w-4 h-4 text-gray-500" />
            <span className="text-gray-600">4 Databases Connected</span>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <Zap className="w-4 h-4 text-green-500" />
            <span className="text-gray-600">Cache: 68% Hit Rate</span>
          </div>
        </div>
      </div>
    </header>
  );
}