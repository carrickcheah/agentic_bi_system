// Mock API Service with realistic business data and delays

export interface Investigation {
  id: string;
  query: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  complexity: 'simple' | 'analytical' | 'computational' | 'investigative';
  phases: InvestigationPhase[];
  insights: Insight[];
  createdAt: Date;
  completedAt?: Date;
  cacheHit: boolean;
  executionTime: number;
}

export interface InvestigationPhase {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed';
  progress: number;
  insights: string[];
}

export interface Insight {
  id: string;
  type: 'finding' | 'recommendation' | 'warning';
  title: string;
  description: string;
  confidence: number;
  data?: any;
}

export interface QueryResult {
  query: string;
  sql: string;
  data: any[];
  executionTime: number;
  rowCount: number;
}

// Mock data generators
const generateSalesData = () => {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return months.map((month, i) => ({
    month,
    sales: Math.floor(Math.random() * 9000000) + 1000000, // 1M to 10M
    target: 8500000, // 8.5M target
    growth: (Math.random() * 0.3 - 0.1), // -10% to +20%
  }));
};

const generateCustomerData = () => {
  return Array.from({ length: 10 }, (_, i) => ({
    id: `CUST-${1000 + i}`,
    name: `Customer ${i + 1}`,
    revenue: Math.floor(Math.random() * 8000000) + 1000000, // 1M to 9M per customer
    satisfaction: Math.floor(Math.random() * 30) + 70,
    churnRisk: Math.random() < 0.3 ? 'high' : Math.random() < 0.6 ? 'medium' : 'low',
  }));
};

// Simulate different investigation scenarios
const investigationScenarios = {
  salesDecline: {
    phases: [
      { name: 'Query Processing', insights: ['Detected Q4 sales analysis request', 'Identified time-series pattern'] },
      { name: 'Strategy Planning', insights: ['Selected trend analysis methodology', 'Including YoY comparison'] },
      { name: 'Data Collection', insights: ['Retrieved 12 months sales data', 'Gathered regional breakdowns'] },
      { name: 'Analysis Execution', insights: ['15% decline in Q4 identified', 'Northeast region most affected'] },
      { name: 'Insight Synthesis', insights: ['Root cause: Competitor pricing', 'Recommendation: Targeted promotions'] },
    ],
    insights: [
      {
        type: 'finding' as const,
        title: 'Q4 Sales Declined 15%',
        description: 'Sales dropped from MYR 4.5M to MYR 3.8M compared to Q3, primarily in Northeast region',
        confidence: 0.92,
      },
      {
        type: 'recommendation' as const,
        title: 'Implement Regional Promotions',
        description: 'Launch targeted 10% discount campaign in Northeast to counter competitor pricing',
        confidence: 0.85,
      },
    ],
  },
  customerChurn: {
    phases: [
      { name: 'Query Processing', insights: ['Customer satisfaction investigation requested', 'Multi-factor analysis needed'] },
      { name: 'Strategy Planning', insights: ['Hypothesis: Product changes impact', 'Cross-reference support tickets'] },
      { name: 'Data Collection', insights: ['6-month satisfaction trends', 'Support ticket analysis'] },
      { name: 'Analysis Execution', insights: ['340% increase in navigation complaints', 'Enterprise segment most affected'] },
      { name: 'Insight Synthesis', insights: ['UI changes causing confusion', 'Immediate action required'] },
    ],
    insights: [
      {
        type: 'warning' as const,
        title: 'Critical: Navigation Issues',
        description: 'Enterprise customers reporting 340% increase in navigation-related support tickets',
        confidence: 0.94,
      },
      {
        type: 'recommendation' as const,
        title: 'Rollback UI Changes',
        description: 'Revert to previous navigation for enterprise accounts while improving onboarding',
        confidence: 0.88,
      },
    ],
  },
};

// Simulate WebSocket-like updates
export const createInvestigationStream = (query: string, onUpdate: (investigation: Investigation) => void) => {
  const scenario = query.toLowerCase().includes('sales') ? 'salesDecline' : 'customerChurn';
  const { phases, insights } = investigationScenarios[scenario];
  
  const investigation: Investigation = {
    id: `INV-${Date.now()}`,
    query,
    status: 'processing',
    complexity: 'investigative',
    phases: phases.map((p, i) => ({
      id: `phase-${i}`,
      name: p.name,
      status: 'pending',
      progress: 0,
      insights: p.insights,
    })),
    insights: [],
    createdAt: new Date(),
    cacheHit: Math.random() > 0.7,
    executionTime: 0,
  };

  onUpdate(investigation);

  // Store intervals for cleanup
  const intervals: NodeJS.Timeout[] = [];
  let currentPhase = 0;
  let isCompleted = false;

  // Simulate phase progression
  const phaseInterval = setInterval(() => {
    if (isCompleted || currentPhase >= investigation.phases.length) {
      investigation.status = 'completed';
      investigation.completedAt = new Date();
      investigation.executionTime = investigation.cacheHit ? 52 : 8432;
      investigation.insights = insights;
      onUpdate({ ...investigation });
      isCompleted = true;
      intervals.forEach(clearInterval);
      return;
    }

    // Update current phase
    if (investigation.phases[currentPhase]) {
      investigation.phases[currentPhase].status = 'in_progress';
      let progress = 0;
      
      const progressInterval = setInterval(() => {
        if (isCompleted || !investigation.phases[currentPhase]) {
          clearInterval(progressInterval);
          return;
        }
        
        progress += 20;
        investigation.phases[currentPhase].progress = Math.min(progress, 100);
        onUpdate({ ...investigation });
        
        if (progress >= 100) {
          investigation.phases[currentPhase].status = 'completed';
          currentPhase++;
          clearInterval(progressInterval);
        }
      }, investigation.cacheHit ? 10 : 300);
      
      intervals.push(progressInterval);
    }
  }, investigation.cacheHit ? 50 : 1500);

  intervals.push(phaseInterval);

  return () => {
    isCompleted = true;
    intervals.forEach(clearInterval);
  };
};

// Mock API endpoints
export const mockApi = {
  // Execute a business query
  executeQuery: async (query: string): Promise<QueryResult> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
    
    return {
      query,
      sql: `SELECT month, sales, target FROM sales_data WHERE year = 2024`,
      data: generateSalesData(),
      executionTime: Math.floor(Math.random() * 2000) + 1000,
      rowCount: 12,
    };
  },

  // Get cached results
  checkCache: async (query: string): Promise<{ hit: boolean; data?: any }> => {
    await new Promise(resolve => setTimeout(resolve, 50));
    
    const hit = Math.random() > 0.3;
    return {
      hit,
      data: hit ? generateSalesData() : undefined,
    };
  },

  // Get investigation history
  getInvestigations: async (): Promise<Investigation[]> => {
    await new Promise(resolve => setTimeout(resolve, 200));
    
    return [
      {
        id: 'INV-001',
        query: 'Why did Q4 sales drop?',
        status: 'completed',
        complexity: 'investigative',
        phases: [],
        insights: investigationScenarios.salesDecline.insights,
        createdAt: new Date(Date.now() - 3600000),
        completedAt: new Date(Date.now() - 3540000),
        cacheHit: false,
        executionTime: 8234,
      },
      {
        id: 'INV-002',
        query: 'What were yesterday\'s sales?',
        status: 'completed',
        complexity: 'simple',
        phases: [],
        insights: [{
          id: '1',
          type: 'finding',
          title: 'Yesterday\'s Sales: MYR 4,783,200',
          description: 'Sales exceeded target by 8% with strong performance in West region',
          confidence: 0.98,
          data: { amount: 4783200, target: 4400000, variance: 0.08, currency: 'MYR' },
        }],
        createdAt: new Date(Date.now() - 7200000),
        completedAt: new Date(Date.now() - 7199950),
        cacheHit: true,
        executionTime: 47,
      },
    ];
  },

  // Export investigation results
  exportResults: async (investigationId: string, format: 'pdf' | 'excel' | 'ppt'): Promise<Blob> => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Return mock blob
    return new Blob(['Mock export data'], { type: 'application/octet-stream' });
  },
};