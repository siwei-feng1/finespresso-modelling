-- Database schema extracted from source database
-- Generated automatically by extract_ddl.py
-- Database: Finespresso

-- Table: fe_logs
CREATE TABLE fe_logs (
    id INTEGER NOT NULL DEFAULT nextval('fe_logs_id_seq'::regclass),
    message VARCHAR NOT NULL,
    timestamp TIMESTAMP,
    status VARCHAR NOT NULL,
    PRIMARY KEY (id)
);


-- Table: eq_model_results_binary
CREATE TABLE eq_model_results_binary (
    id INTEGER NOT NULL DEFAULT nextval('eq_model_results_binary_id_seq'::regclass),
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event VARCHAR(255) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    auc_roc DOUBLE PRECISION,
    test_sample INTEGER NOT NULL,
    training_sample INTEGER NOT NULL,
    total_sample INTEGER NOT NULL,
    up_accuracy DOUBLE PRECISION,
    down_accuracy DOUBLE PRECISION,
    total_up INTEGER,
    total_down INTEGER,
    correct_up INTEGER,
    correct_down INTEGER,
    up_predictions_pct DOUBLE PRECISION,
    down_predictions_pct DOUBLE PRECISION,
    PRIMARY KEY (id)
);


-- Table: eq_model_results_regression
CREATE TABLE eq_model_results_regression (
    id INTEGER NOT NULL DEFAULT nextval('eq_model_results_regression_id_seq'::regclass),
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event VARCHAR(255) NOT NULL,
    mse DOUBLE PRECISION NOT NULL,
    r2 DOUBLE PRECISION NOT NULL,
    mae DOUBLE PRECISION NOT NULL,
    rmse DOUBLE PRECISION NOT NULL,
    test_sample INTEGER NOT NULL,
    training_sample INTEGER NOT NULL,
    total_sample INTEGER NOT NULL,
    PRIMARY KEY (id)
);


-- Table: instrument
CREATE TABLE instrument (
    id BIGINT NOT NULL DEFAULT nextval('instrument_id_seq'::regclass),
    issuer VARCHAR(255),
    ticker VARCHAR(100),
    yf_ticker VARCHAR(100),
    isin VARCHAR(100),
    asset_class VARCHAR(100),
    exchange VARCHAR(100),
    exchange_code VARCHAR(100),
    country VARCHAR(100),
    url TEXT,
    sector VARCHAR(100),
    market_cap_class VARCHAR(100),
    float_ratio DOUBLE PRECISION,
    PRIMARY KEY (id)
);


-- Table: news
CREATE TABLE news (
    id INTEGER NOT NULL DEFAULT nextval('news_id_seq'::regclass),
    title TEXT,
    link TEXT,
    company TEXT,
    published_date TIMESTAMP,
    content TEXT,
    reason TEXT,
    industry TEXT,
    publisher_topic TEXT,
    event VARCHAR(255),
    publisher VARCHAR(255),
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(255),
    instrument_id BIGINT,
    yf_ticker VARCHAR(255),
    ticker VARCHAR(16),
    published_date_gmt TIMESTAMP,
    timezone VARCHAR(50),
    publisher_summary TEXT,
    ticker_url VARCHAR(500),
    predicted_side VARCHAR(10),
    predicted_move DOUBLE PRECISION,
    langauge VARCHAR(10),
    language VARCHAR(50),
    content_en TEXT,
    title_en TEXT,
    PRIMARY KEY (id)
);


-- Table: price_moves
CREATE TABLE price_moves (
    id INTEGER NOT NULL DEFAULT nextval('price_moves_id_seq'::regclass),
    news_id INTEGER NOT NULL,
    ticker VARCHAR NOT NULL,
    published_date TIMESTAMP NOT NULL,
    begin_price DOUBLE PRECISION NOT NULL,
    end_price DOUBLE PRECISION NOT NULL,
    index_begin_price DOUBLE PRECISION NOT NULL,
    index_end_price DOUBLE PRECISION NOT NULL,
    volume INTEGER NOT NULL,
    market VARCHAR NOT NULL,
    price_change DOUBLE PRECISION NOT NULL,
    price_change_percentage DOUBLE PRECISION,
    index_price_change DOUBLE PRECISION NOT NULL,
    index_price_change_percentage DOUBLE PRECISION NOT NULL,
    daily_alpha DOUBLE PRECISION NOT NULL,
    actual_side VARCHAR(10) NOT NULL,
    predicted_side VARCHAR(10),
    predicted_move DOUBLE PRECISION,
    downloaded_at TIMESTAMP,
    price_source VARCHAR(20) NOT NULL DEFAULT 'yfinance'::character varying,
    PRIMARY KEY (id)
);


-- Table: signups
CREATE TABLE signups (
    id INTEGER NOT NULL DEFAULT nextval('signups_id_seq'::regclass),
    email VARCHAR(255) NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);


-- Table: conversations
CREATE TABLE conversations (
    id INTEGER NOT NULL DEFAULT nextval('conversations_id_seq'::regclass),
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_prompt TEXT,
    answer TEXT,
    PRIMARY KEY (id)
);


