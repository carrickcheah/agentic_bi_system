-- Pattern Statistics Database Schema
-- Creates tables for tracking business intelligence pattern usage and success rates

-- Table for storing pattern statistics and success rates
CREATE TABLE IF NOT EXISTS pattern_statistics (
    pattern_id VARCHAR(50) PRIMARY KEY,
    success_rate FLOAT NOT NULL DEFAULT 0.55,
    confidence_lower FLOAT NOT NULL DEFAULT 0.45,
    confidence_upper FLOAT NOT NULL DEFAULT 0.65,
    data_source VARCHAR(30) NOT NULL DEFAULT 'bootstrap_estimate',
    sample_size INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    usage_count INTEGER NOT NULL DEFAULT 0,
    average_investigation_time FLOAT,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing detailed investigation outcomes
CREATE TABLE IF NOT EXISTS investigation_outcomes (
    id SERIAL PRIMARY KEY,
    pattern_id VARCHAR(50) NOT NULL,
    investigation_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    completion_success BOOLEAN NOT NULL,
    user_satisfaction_score FLOAT NOT NULL CHECK (user_satisfaction_score >= 0.0 AND user_satisfaction_score <= 1.0),
    accuracy_validation BOOLEAN NOT NULL,
    implementation_success BOOLEAN,
    investigation_time_minutes FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (pattern_id) REFERENCES pattern_statistics(pattern_id) ON DELETE CASCADE
);

-- Table for tracking pattern usage frequency by user and domain
CREATE TABLE IF NOT EXISTS pattern_usage_analytics (
    id SERIAL PRIMARY KEY,
    pattern_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(100),
    business_domain VARCHAR(50) NOT NULL,
    usage_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    match_confidence FLOAT,
    query_text TEXT,
    
    FOREIGN KEY (pattern_id) REFERENCES pattern_statistics(pattern_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_pattern_statistics_data_source ON pattern_statistics(data_source);
CREATE INDEX IF NOT EXISTS idx_pattern_statistics_last_updated ON pattern_statistics(last_updated);
CREATE INDEX IF NOT EXISTS idx_pattern_statistics_success_rate ON pattern_statistics(success_rate);

CREATE INDEX IF NOT EXISTS idx_investigation_outcomes_pattern_id ON investigation_outcomes(pattern_id);
CREATE INDEX IF NOT EXISTS idx_investigation_outcomes_timestamp ON investigation_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_investigation_outcomes_user_id ON investigation_outcomes(user_id);

CREATE INDEX IF NOT EXISTS idx_pattern_usage_analytics_pattern_id ON pattern_usage_analytics(pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_usage_analytics_user_id ON pattern_usage_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_pattern_usage_analytics_business_domain ON pattern_usage_analytics(business_domain);
CREATE INDEX IF NOT EXISTS idx_pattern_usage_analytics_timestamp ON pattern_usage_analytics(usage_timestamp);

-- Create a view for pattern effectiveness summary
CREATE OR REPLACE VIEW pattern_effectiveness_summary AS
SELECT 
    ps.pattern_id,
    ps.success_rate,
    ps.confidence_lower,
    ps.confidence_upper,
    ps.data_source,
    ps.sample_size,
    ps.usage_count,
    ps.average_investigation_time,
    COUNT(io.id) as total_investigations,
    AVG(CASE WHEN io.completion_success THEN 1.0 ELSE 0.0 END) as actual_completion_rate,
    AVG(io.user_satisfaction_score) as avg_user_satisfaction,
    AVG(CASE WHEN io.accuracy_validation THEN 1.0 ELSE 0.0 END) as actual_accuracy_rate,
    AVG(io.investigation_time_minutes) as avg_actual_investigation_time,
    ps.last_updated
FROM pattern_statistics ps
LEFT JOIN investigation_outcomes io ON ps.pattern_id = io.pattern_id
WHERE ps.active = true
GROUP BY ps.pattern_id, ps.success_rate, ps.confidence_lower, ps.confidence_upper, 
         ps.data_source, ps.sample_size, ps.usage_count, ps.average_investigation_time, ps.last_updated;

-- Create a view for business domain analytics
CREATE OR REPLACE VIEW domain_pattern_analytics AS
SELECT 
    pua.business_domain,
    COUNT(DISTINCT pua.pattern_id) as unique_patterns_used,
    COUNT(*) as total_usage,
    COUNT(DISTINCT pua.user_id) as unique_users,
    AVG(pua.match_confidence) as avg_match_confidence,
    AVG(ps.success_rate) as avg_domain_success_rate
FROM pattern_usage_analytics pua
JOIN pattern_statistics ps ON pua.pattern_id = ps.pattern_id
WHERE ps.active = true
GROUP BY pua.business_domain;

-- Create a view for user expertise analytics
CREATE OR REPLACE VIEW user_expertise_analytics AS
SELECT 
    pua.user_id,
    pua.user_role,
    COUNT(DISTINCT pua.pattern_id) as patterns_used,
    COUNT(*) as total_usage,
    COUNT(DISTINCT pua.business_domain) as domains_explored,
    AVG(pua.match_confidence) as avg_match_confidence,
    AVG(io.user_satisfaction_score) as avg_satisfaction,
    AVG(CASE WHEN io.completion_success THEN 1.0 ELSE 0.0 END) as success_rate
FROM pattern_usage_analytics pua
LEFT JOIN investigation_outcomes io ON pua.pattern_id = io.pattern_id AND pua.user_id = io.user_id
GROUP BY pua.user_id, pua.user_role;

-- Insert sample bootstrap data for all 150 patterns
-- This will be populated by the pattern library initialization

COMMENT ON TABLE pattern_statistics IS 'Stores success rates and usage statistics for business intelligence patterns';
COMMENT ON TABLE investigation_outcomes IS 'Tracks detailed outcomes of pattern-guided investigations for learning';
COMMENT ON TABLE pattern_usage_analytics IS 'Analytics for pattern usage patterns and user behavior';
COMMENT ON VIEW pattern_effectiveness_summary IS 'Summary view of pattern effectiveness with actual vs predicted metrics';
COMMENT ON VIEW domain_pattern_analytics IS 'Analytics for pattern usage by business domain';
COMMENT ON VIEW user_expertise_analytics IS 'Analytics for user expertise and pattern usage patterns';