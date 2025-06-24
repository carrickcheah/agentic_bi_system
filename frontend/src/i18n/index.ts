import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Translation resources
const resources = {
  en: {
    translation: {
      // Header
      "agentic_bi": "Agentic BI",
      "world_class_intelligence": "World-Class Autonomous Intelligence",
      
      // Sidebar
      "new_analysis": "New Analysis",
      "reports": "Reports", 
      "projects": "Projects",
      "recents": "Recents",
      
      // User Menu
      "settings": "Settings",
      "language": "Language",
      "get_help": "Get help",
      "dark_mode": "Dark mode",
      "learn_more": "Learn more",
      "log_out": "Log out",
      "personal": "Personal",
      "pro_plan": "Pro Plan",
      
      // Conversation
      "welcome_message": "Hi! How can i help you today?",
      "send": "Send",
      "investigating": "Investigating",
      "powered_by": "Powered by Claude Sonnet 4.0",
      "thinking_mode": "Thinking mode",
      "enter_to_send": "to send",
      "ask_placeholder": "Ask me anything about your business intelligence...",
      
      // Suggested queries
      "try_queries": "Try these powerful queries:",
      "q4_sales_drop": "Why did Q4 sales drop?",
      "yesterday_sales": "What were yesterday's sales?",
      "satisfaction_trends": "Customer satisfaction trends", 
      "revenue_forecast": "Revenue forecast next quarter",
      
      // Recent chats
      "agentic_sql_research": "Agentic SQL Agent Research",
      "q4_sales_analysis": "Q4 Sales Performance Analysis",
      "customer_segmentation": "Customer Segmentation Study",
      "revenue_forecast_model": "Revenue Forecast Model",
      "product_analytics": "Product Analytics Dashboard",
      "market_trend": "Market Trend Analysis",
      "operational_efficiency": "Operational Efficiency Report",
      "competitive_analysis": "Competitive Analysis Framework",
      "user_behavior": "User Behavior Insights",
      "financial_performance": "Financial Performance Review",
      "supply_chain": "Supply Chain Optimization",
      "risk_assessment": "Risk Assessment Report"
    }
  },
  ms: {
    translation: {
      // Header
      "agentic_bi": "Agentic BI",
      "world_class_intelligence": "Kecerdasan Autonomi Bertaraf Dunia",
      
      // Sidebar
      "new_analysis": "Analisis Baharu",
      "reports": "Laporan",
      "projects": "Projek",
      "recents": "Terkini",
      
      // User Menu
      "settings": "Tetapan",
      "language": "Bahasa",
      "get_help": "Dapatkan bantuan",
      "dark_mode": "Mod gelap",
      "learn_more": "Ketahui lebih lanjut",
      "log_out": "Log keluar",
      "personal": "Peribadi",
      "pro_plan": "Pelan Pro",
      
      // Conversation
      "welcome_message": "Hai! Apa yang boleh saya bantu anda hari ini?",
      "send": "Hantar",
      "investigating": "Menyiasat",
      "powered_by": "Dikuasakan oleh Claude Sonnet 4.0",
      "thinking_mode": "Mod berfikir",
      "enter_to_send": "untuk hantar",
      "ask_placeholder": "Tanya saya apa-apa tentang kecerdasan perniagaan anda...",
      
      // Suggested queries
      "try_queries": "Cuba pertanyaan berkuasa ini:",
      "q4_sales_drop": "Mengapa jualan Q4 menurun?",
      "yesterday_sales": "Berapa jualan semalam?",
      "satisfaction_trends": "Trend kepuasan pelanggan",
      "revenue_forecast": "Ramalan hasil suku seterusnya",
      
      // Recent chats
      "agentic_sql_research": "Penyelidikan Agen SQL Agentic",
      "q4_sales_analysis": "Analisis Prestasi Jualan Q4",
      "customer_segmentation": "Kajian Segmentasi Pelanggan",
      "revenue_forecast_model": "Model Ramalan Hasil",
      "product_analytics": "Dashboard Analitik Produk",
      "market_trend": "Analisis Trend Pasaran",
      "operational_efficiency": "Laporan Kecekapan Operasi",
      "competitive_analysis": "Rangka Kerja Analisis Persaingan",
      "user_behavior": "Pandangan Tingkah Laku Pengguna",
      "financial_performance": "Ulasan Prestasi Kewangan",
      "supply_chain": "Pengoptimuman Rantai Bekalan",
      "risk_assessment": "Laporan Penilaian Risiko"
    }
  },
  zh: {
    translation: {
      // Header
      "agentic_bi": "Agentic BI",
      "world_class_intelligence": "世界级自主智能",
      
      // Sidebar
      "new_analysis": "新分析",
      "reports": "报告",
      "projects": "项目",
      "recents": "最近",
      
      // User Menu
      "settings": "设置",
      "language": "语言",
      "get_help": "获取帮助",
      "dark_mode": "深色模式",
      "learn_more": "了解更多",
      "log_out": "登出",
      "personal": "个人",
      "pro_plan": "专业版",
      
      // Conversation
      "welcome_message": "您好！今天我可以为您做些什么？",
      "send": "发送",
      "investigating": "调查中",
      "powered_by": "由 Claude Sonnet 4.0 提供支持",
      "thinking_mode": "思考模式",
      "enter_to_send": "发送",
      "ask_placeholder": "询问我任何关于您的商业智能的问题...",
      
      // Suggested queries
      "try_queries": "尝试这些强大的查询：",
      "q4_sales_drop": "为什么第四季度销售下降？",
      "yesterday_sales": "昨天的销售额是多少？",
      "satisfaction_trends": "客户满意度趋势",
      "revenue_forecast": "下季度收入预测",
      
      // Recent chats
      "agentic_sql_research": "Agentic SQL 代理研究",
      "q4_sales_analysis": "第四季度销售绩效分析",
      "customer_segmentation": "客户细分研究",
      "revenue_forecast_model": "收入预测模型",
      "product_analytics": "产品分析仪表板",
      "market_trend": "市场趋势分析",
      "operational_efficiency": "运营效率报告",
      "competitive_analysis": "竞争分析框架",
      "user_behavior": "用户行为洞察",
      "financial_performance": "财务绩效审查",
      "supply_chain": "供应链优化",
      "risk_assessment": "风险评估报告"
    }
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    debug: false,
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
    },
  });

export default i18n;