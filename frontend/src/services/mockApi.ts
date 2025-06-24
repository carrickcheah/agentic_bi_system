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
      { name: 'Data Source Discovery', insights: ['Connected to MariaDB sales database', 'Found 247K transaction records'] },
      { name: 'Strategy Planning', insights: ['Selected trend analysis methodology', 'Including YoY comparison'] },
      { name: 'Data Collection', insights: ['Retrieved 12 months sales data', 'Gathered regional breakdowns'] },
      { name: 'Pattern Recognition', insights: ['Applied ML algorithms', 'Detected seasonal anomalies'] },
      { name: 'Cross-Reference Analysis', insights: ['Analyzed competitor pricing data', 'Correlated with market events'] },
      { name: 'Analysis Execution', insights: ['15% decline in Q4 identified', 'Northeast region most affected'] },
      { name: 'Insight Synthesis', insights: ['Root cause: Competitor pricing', 'Recommendation: Targeted promotions'] },
    ],
    insights: [
      {
        id: 'sales-finding-1',
        type: 'finding' as const,
        title: 'Q4 Sales Declined 15%',
        description: 'Sales dropped from MYR 4.5M to MYR 3.8M compared to Q3, primarily in Northeast region',
        confidence: 0.92,
      },
      {
        id: 'sales-rec-1',
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
      { name: 'Data Source Discovery', insights: ['Connected to PostgreSQL CRM', 'Accessed Qdrant vector database'] },
      { name: 'Strategy Planning', insights: ['Hypothesis: Product changes impact', 'Cross-reference support tickets'] },
      { name: 'Data Collection', insights: ['6-month satisfaction trends', 'Support ticket analysis'] },
      { name: 'Sentiment Analysis', insights: ['Processing customer feedback', 'NLP analysis on 15K reviews'] },
      { name: 'Behavioral Modeling', insights: ['User journey mapping completed', 'Identified friction points'] },
      { name: 'Analysis Execution', insights: ['340% increase in navigation complaints', 'Enterprise segment most affected'] },
      { name: 'Insight Synthesis', insights: ['UI changes causing confusion', 'Immediate action required'] },
    ],
    insights: [
      {
        id: 'churn-warning-1',
        type: 'warning' as const,
        title: 'Critical: Navigation Issues',
        description: 'Enterprise customers reporting 340% increase in navigation-related support tickets',
        confidence: 0.94,
      },
      {
        id: 'churn-rec-1',
        type: 'recommendation' as const,
        title: 'Rollback UI Changes',
        description: 'Revert to previous navigation for enterprise accounts while improving onboarding',
        confidence: 0.88,
      },
    ],
  },
  revenueForecast: {
    phases: [
      { name: 'Query Processing', insights: ['Revenue forecast analysis requested', 'Q1 2024 projection scope'] },
      { name: 'Data Source Discovery', insights: ['Connected to financial database', 'Retrieved 3-year revenue history'] },
      { name: 'Strategy Planning', insights: ['Selected trend analysis methodology', 'Including seasonal patterns'] },
      { name: 'Historical Analysis', insights: ['Analyzed revenue patterns', 'Identified growth trends and seasonality'] },
      { name: 'Market Research', insights: ['Reviewed industry benchmarks', 'Assessed economic indicators impact'] },
      { name: 'Scenario Planning', insights: ['Developed optimistic/conservative projections', 'Created risk-adjusted forecasts'] },
      { name: 'Analysis Execution', insights: ['Revenue projections completed', 'Confidence ranges established'] },
      { name: 'Insight Synthesis', insights: ['23% growth projected', 'Key risk factors identified'] },
    ],
    insights: [
      {
        id: 'revenue-finding-1',
        type: 'finding' as const,
        title: 'Q1 2024 Revenue: MYR 5.8M',
        description: 'Projected 23% growth with 85% confidence interval of MYR 5.2M - 6.4M',
        confidence: 0.89,
      },
      {
        id: 'revenue-rec-1',
        type: 'recommendation' as const,
        title: 'Accelerate Enterprise Sales',
        description: 'Focus on enterprise segment which shows highest growth potential for Q1',
        confidence: 0.91,
      },
    ],
  },
  productAnalytics: {
    phases: [
      { name: 'Query Processing', insights: ['Product performance analysis requested', 'Multi-dimensional analysis scope'] },
      { name: 'Data Source Discovery', insights: ['Connected to product database', 'Found 2.3M user interactions'] },
      { name: 'Strategy Planning', insights: ['Selected customer behavior analysis', 'Including feature adoption metrics'] },
      { name: 'Customer Segmentation', insights: ['Identified 7 distinct customer segments', 'Premium users represent 12% of base'] },
      { name: 'Usage Pattern Analysis', insights: ['Analyzed feature usage patterns', 'Mapped customer journey touchpoints'] },
      { name: 'Performance Evaluation', insights: ['Reviewed conversion metrics', 'Identified 3 major drop-off points'] },
      { name: 'Analysis Execution', insights: ['Product metrics calculated', 'Customer retention analyzed'] },
      { name: 'Insight Synthesis', insights: ['Premium feature drives 34% more engagement', 'Onboarding improvements needed'] },
    ],
    insights: [
      {
        id: 'product-finding-1',
        type: 'finding' as const,
        title: 'Premium Feature Increases Engagement 34%',
        description: 'Customers who use premium features show 34% higher monthly engagement and 28% better retention',
        confidence: 0.93,
      },
      {
        id: 'product-rec-1',
        type: 'recommendation' as const,
        title: 'Improve Onboarding Flow',
        description: 'Redesign onboarding to highlight Feature X and reduce 47% drop-off at step 3',
        confidence: 0.87,
      },
    ],
  },
  marketTrends: {
    phases: [
      { name: 'Query Processing', insights: ['Market trend analysis requested', 'Competitive landscape scope'] },
      { name: 'Data Source Discovery', insights: ['Connected to market research database', 'Integrated industry reports'] },
      { name: 'Strategy Planning', insights: ['Selected competitive analysis framework', 'Including market positioning review'] },
      { name: 'Competitive Research', insights: ['Analyzed 12 competitor strategies', 'Reviewed pricing and positioning'] },
      { name: 'Market Intelligence', insights: ['Processed industry reports', 'Brand perception trending positive'] },
      { name: 'Opportunity Assessment', insights: ['Emerging market opportunities identified', 'Growth potential indicators found'] },
      { name: 'Analysis Execution', insights: ['Market share analysis complete', 'Opportunity evaluation ready'] },
      { name: 'Insight Synthesis', insights: ['Untapped market segment found', 'Competitive advantage identified'] },
    ],
    insights: [
      {
        id: 'market-finding-1',
        type: 'finding' as const,
        title: 'Untapped SMB Market Worth MYR 12M',
        description: 'Small-medium business segment shows 67% unmet demand in our service category',
        confidence: 0.88,
      },
      {
        id: 'market-rec-1',
        type: 'recommendation' as const,
        title: 'Launch SMB-Focused Product Line',
        description: 'Develop simplified product variant targeting SMB segment with 40% price reduction',
        confidence: 0.82,
      },
    ],
  },
};

// Simulate WebSocket-like updates
export const createInvestigationStream = (query: string, onUpdate: (investigation: Investigation) => void) => {
  // Smart scenario detection based on query content
  let scenario = 'customerChurn'; // default
  
  const queryLower = query.toLowerCase();
  if (queryLower.includes('sales') || queryLower.includes('drop') || queryLower.includes('decline')) {
    scenario = 'salesDecline';
  } else if (queryLower.includes('revenue') || queryLower.includes('forecast') || queryLower.includes('predict')) {
    scenario = 'revenueForecast';
  } else if (queryLower.includes('product') || queryLower.includes('feature') || queryLower.includes('analytics')) {
    scenario = 'productAnalytics';
  } else if (queryLower.includes('market') || queryLower.includes('trend') || queryLower.includes('competitor')) {
    scenario = 'marketTrends';
  } else if (queryLower.includes('satisfaction') || queryLower.includes('churn') || queryLower.includes('customer')) {
    scenario = 'customerChurn';
  }
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

  // Simulate phase progression with 1-second delays
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
      }, investigation.cacheHit ? 50 : 200); // Faster progress within each phase
      
      intervals.push(progressInterval);
    }
  }, 2000); // 2 second delay between each phase

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