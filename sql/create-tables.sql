CREATE TABLE news (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    link VARCHAR(255),
    company VARCHAR(255),
    published_date TIMESTAMP WITH TIME ZONE,
    content TEXT,
    ai_summary TEXT,
    industry VARCHAR(255),
    publisher_topic VARCHAR(255),
    ai_topic VARCHAR(255),
    publisher VARCHAR(255)
);


CREATE TABLE company (
    id SERIAL PRIMARY KEY,
    yf_ticker VARCHAR(255),
    mw_ticker VARCHAR(255),
    yf_url VARCHAR(255),
    mw_url VARCHAR(255)
);

CREATE TABLE price_moves (
    id SERIAL PRIMARY KEY,
    news_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    published_date TIMESTAMP NOT NULL,
    begin_price FLOAT NOT NULL,
    end_price FLOAT NOT NULL,
    index_begin_price FLOAT NOT NULL,
    index_end_price FLOAT NOT NULL,
    volume INTEGER NOT NULL,
    market VARCHAR NOT NULL,
    price_change FLOAT NOT NULL,
    price_change_percentage FLOAT,
    index_price_change FLOAT NOT NULL,
    index_price_change_percentage FLOAT NOT NULL,
    daily_alpha FLOAT NOT NULL,
    actual_side VARCHAR(10) NOT NULL,
    predicted_side VARCHAR(10),
    predicted_move FLOAT
);

-- Create eq_model_results_binary table
CREATE TABLE eq_model_results_binary (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event VARCHAR(255) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    test_sample INTEGER NOT NULL,
    training_sample INTEGER NOT NULL,
    total_sample INTEGER NOT NULL,
    up_accuracy FLOAT,
    down_accuracy FLOAT,
    total_up INTEGER,
    total_down INTEGER,
    correct_up INTEGER,
    correct_down INTEGER,
    up_predictions_pct FLOAT,
    down_predictions_pct FLOAT
);

-- Create eq_model_results_regression table
CREATE TABLE eq_model_results_regression (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event VARCHAR(255) NOT NULL,
    mse FLOAT NOT NULL,
    r2 FLOAT NOT NULL,
    mae FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    test_sample INTEGER NOT NULL,
    training_sample INTEGER NOT NULL,
    total_sample INTEGER NOT NULL
);

-- Create models table to store model binaries
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'classifier_binary', 'regression', 'vectorizer'
    event VARCHAR(255) NOT NULL,
    model_binary BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(name, version)
);