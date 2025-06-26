-- =====================================================
-- Semantic Cache Table for Organization-wide Pattern Matching
-- =====================================================
-- This table stores business intelligence patterns for semantic similarity matching
-- and organizational knowledge accumulation with indefinite TTL.

-- Create semantic_cache table
CREATE TABLE IF NOT EXISTS semantic_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Semantic Pattern Identification
    pattern_id VARCHAR(64) NOT NULL UNIQUE,
    semantic_hash VARCHAR(64) NOT NULL,
    
    -- Business Context
    organization_id VARCHAR(255) NOT NULL,
    business_domain VARCHAR(100) NOT NULL,
    
    -- Semantic Intent and Insights
    semantic_intent JSONB NOT NULL,
    insights JSONB NOT NULL,
    
    -- Pattern Metadata
    original_question TEXT,
    question_type VARCHAR(100),
    analysis_type VARCHAR(100),
    
    -- Access Control (Pattern-based, not user-specific)
    required_permissions JSONB DEFAULT '[]'::jsonb,
    public_pattern BOOLEAN DEFAULT false,
    
    -- Usage Analytics
    usage_count INTEGER DEFAULT 0,
    similarity_threshold DOUBLE PRECISION DEFAULT 0.75,
    
    -- Timestamps (Indefinite TTL - no expires_at)
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Pattern Status
    active BOOLEAN DEFAULT true,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    
    -- Additional Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Vector Embedding (for future vector similarity)
    embedding_vector FLOAT[] -- For pgvector extension if available
);

-- =====================================================
-- Indexes for Semantic Cache Performance
-- =====================================================

-- Primary lookup indexes
CREATE INDEX IF NOT EXISTS idx_semantic_cache_pattern_id 
    ON semantic_cache(pattern_id);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_semantic_hash 
    ON semantic_cache(semantic_hash);

-- Organizational and domain filtering
CREATE INDEX IF NOT EXISTS idx_semantic_cache_org_domain 
    ON semantic_cache(organization_id, business_domain);

-- Business context indexes
CREATE INDEX IF NOT EXISTS idx_semantic_cache_business_domain 
    ON semantic_cache(business_domain);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_question_type 
    ON semantic_cache(question_type);

-- Usage and analytics indexes
CREATE INDEX IF NOT EXISTS idx_semantic_cache_usage_count 
    ON semantic_cache(usage_count DESC);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_last_used 
    ON semantic_cache(last_used DESC);

-- Status and filtering indexes
CREATE INDEX IF NOT EXISTS idx_semantic_cache_active 
    ON semantic_cache(active) WHERE active = true;

CREATE INDEX IF NOT EXISTS idx_semantic_cache_public 
    ON semantic_cache(public_pattern) WHERE public_pattern = true;

-- JSONB indexes for semantic intent and permissions
CREATE INDEX IF NOT EXISTS idx_semantic_cache_semantic_intent 
    ON semantic_cache USING GIN (semantic_intent);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_permissions 
    ON semantic_cache USING GIN (required_permissions);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_metadata 
    ON semantic_cache USING GIN (metadata);

-- =====================================================
-- Triggers for Semantic Cache
-- =====================================================

-- Function to update usage statistics
CREATE OR REPLACE FUNCTION update_semantic_pattern_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Increment usage count and update last_used when pattern is accessed
    NEW.usage_count = OLD.usage_count + 1;
    NEW.last_used = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update usage when pattern is accessed
CREATE TRIGGER update_semantic_usage
    BEFORE UPDATE ON semantic_cache
    FOR EACH ROW 
    WHEN (OLD.last_used IS DISTINCT FROM NEW.last_used)
    EXECUTE FUNCTION update_semantic_pattern_usage();

-- =====================================================
-- Semantic Cache Helper Functions
-- =====================================================

-- Function to find similar semantic patterns
CREATE OR REPLACE FUNCTION find_similar_semantic_patterns(
    p_organization_id VARCHAR(255),
    p_semantic_intent JSONB,
    p_business_domain VARCHAR(100) DEFAULT NULL,
    p_user_permissions JSONB DEFAULT '[]'::jsonb,
    p_similarity_threshold DOUBLE PRECISION DEFAULT 0.75,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    pattern_id VARCHAR(64),
    semantic_hash VARCHAR(64),
    business_domain VARCHAR(100),
    insights JSONB,
    usage_count INTEGER,
    similarity_score DOUBLE PRECISION,
    last_used TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sc.pattern_id,
        sc.semantic_hash,
        sc.business_domain,
        sc.insights,
        sc.usage_count,
        -- Simple JSON similarity (in production, use vector similarity)
        CASE 
            WHEN sc.semantic_intent = p_semantic_intent THEN 1.0
            WHEN sc.semantic_intent->>'business_domain' = p_semantic_intent->>'business_domain' THEN 0.8
            ELSE 0.5
        END AS similarity_score,
        sc.last_used
    FROM semantic_cache sc
    WHERE sc.organization_id = p_organization_id
        AND sc.active = true
        AND (p_business_domain IS NULL OR sc.business_domain = p_business_domain)
        AND (
            sc.public_pattern = true OR
            sc.required_permissions = '[]'::jsonb OR
            sc.required_permissions ?| ARRAY(SELECT jsonb_array_elements_text(p_user_permissions))
        )
        AND CASE 
            WHEN sc.semantic_intent = p_semantic_intent THEN 1.0
            WHEN sc.semantic_intent->>'business_domain' = p_semantic_intent->>'business_domain' THEN 0.8
            ELSE 0.5
        END >= p_similarity_threshold
    ORDER BY similarity_score DESC, sc.usage_count DESC, sc.last_used DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get popular semantic patterns
CREATE OR REPLACE FUNCTION get_popular_semantic_patterns(
    p_organization_id VARCHAR(255),
    p_business_domain VARCHAR(100) DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    pattern_id VARCHAR(64),
    business_domain VARCHAR(100),
    question_type VARCHAR(100),
    original_question TEXT,
    usage_count INTEGER,
    last_used TIMESTAMP WITH TIME ZONE,
    insights_summary TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sc.pattern_id,
        sc.business_domain,
        sc.question_type,
        sc.original_question,
        sc.usage_count,
        sc.last_used,
        COALESCE(sc.insights->>'summary', 'No summary available') AS insights_summary
    FROM semantic_cache sc
    WHERE sc.organization_id = p_organization_id
        AND sc.active = true
        AND (p_business_domain IS NULL OR sc.business_domain = p_business_domain)
    ORDER BY sc.usage_count DESC, sc.last_used DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old unused patterns (optional maintenance)
CREATE OR REPLACE FUNCTION cleanup_unused_semantic_patterns(
    p_days_threshold INTEGER DEFAULT 90,
    p_usage_threshold INTEGER DEFAULT 0
)
RETURNS INTEGER AS $$
DECLARE
    patterns_deactivated INTEGER := 0;
BEGIN
    -- Deactivate patterns that haven't been used in X days and have low usage
    UPDATE semantic_cache 
    SET active = false, deactivated_at = NOW()
    WHERE active = true
        AND last_used < NOW() - INTERVAL '1 day' * p_days_threshold
        AND usage_count <= p_usage_threshold;
    
    GET DIAGNOSTICS patterns_deactivated = ROW_COUNT;
    
    RAISE NOTICE 'Deactivated % unused semantic patterns', patterns_deactivated;
    
    RETURN patterns_deactivated;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Semantic Cache Statistics Function
-- =====================================================

CREATE OR REPLACE FUNCTION get_semantic_cache_stats(
    p_organization_id VARCHAR(255) DEFAULT NULL
)
RETURNS TABLE (
    total_patterns BIGINT,
    active_patterns BIGINT,
    total_usage BIGINT,
    avg_usage_per_pattern NUMERIC,
    top_business_domain VARCHAR(100),
    most_popular_pattern VARCHAR(64),
    cache_efficiency NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_patterns,
        COUNT(*) FILTER (WHERE active = true) as active_patterns,
        SUM(usage_count) as total_usage,
        ROUND(AVG(usage_count), 2) as avg_usage_per_pattern,
        MODE() WITHIN GROUP (ORDER BY business_domain) as top_business_domain,
        (SELECT pattern_id FROM semantic_cache 
         WHERE (p_organization_id IS NULL OR organization_id = p_organization_id)
         ORDER BY usage_count DESC LIMIT 1) as most_popular_pattern,
        ROUND(
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    (COUNT(*) FILTER (WHERE usage_count > 0)::NUMERIC / COUNT(*)::NUMERIC) * 100
                ELSE 0 
            END, 2
        ) as cache_efficiency
    FROM semantic_cache sc
    WHERE (p_organization_id IS NULL OR sc.organization_id = p_organization_id);
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Sample Data for Testing (Optional)
-- =====================================================

-- Insert some sample semantic patterns for testing
INSERT INTO semantic_cache (
    pattern_id, semantic_hash, organization_id, business_domain,
    semantic_intent, insights, original_question, question_type,
    required_permissions, public_pattern, usage_count
) VALUES
(
    'pattern_sales_monthly_01',
    'hash_sales_monthly_performance',
    'acme_corp',
    'sales',
    '{"business_domain": "sales", "business_intent": {"question_type": "analytical", "time_period": "monthly"}, "analysis_type": "performance"}',
    '{"summary": "Monthly sales analysis pattern", "key_findings": ["Sales trend identification", "Performance metrics"], "pattern_type": "recurring_analysis"}',
    'What is our monthly sales performance?',
    'analytical',
    '["sales_read", "manager"]',
    false,
    15
),
(
    'pattern_finance_dashboard_01', 
    'hash_finance_realtime_metrics',
    'acme_corp',
    'finance',
    '{"business_domain": "finance", "business_intent": {"question_type": "descriptive", "time_period": "current"}, "analysis_type": "dashboard"}',
    '{"summary": "Real-time finance dashboard pattern", "key_findings": ["Cash flow monitoring", "Budget tracking"], "pattern_type": "dashboard_view"}',
    'Show me the current financial dashboard',
    'descriptive',
    '["finance_read"]',
    false,
    32
),
(
    'pattern_customer_support_01',
    'hash_customer_support_tickets',
    'acme_corp', 
    'customer',
    '{"business_domain": "customer", "business_intent": {"question_type": "operational", "time_period": "today"}, "analysis_type": "monitoring"}',
    '{"summary": "Daily customer support monitoring", "key_findings": ["Ticket volume", "Response times"], "pattern_type": "operational_monitoring"}',
    'How many support tickets do we have today?',
    'operational',
    '["support_read", "customer_read"]',
    true,
    8
) ON CONFLICT (pattern_id) DO NOTHING;

-- =====================================================
-- Verification Queries
-- =====================================================

-- Check if table was created successfully
SELECT 'semantic_cache table created successfully' as status
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_name = 'semantic_cache' AND table_schema = 'public'
);

-- Show table structure
\d semantic_cache

-- Show sample data
SELECT pattern_id, business_domain, question_type, usage_count, public_pattern
FROM semantic_cache 
ORDER BY usage_count DESC;

-- Test the helper functions
SELECT * FROM get_semantic_cache_stats('acme_corp');

SELECT * FROM get_popular_semantic_patterns('acme_corp', NULL, 5);