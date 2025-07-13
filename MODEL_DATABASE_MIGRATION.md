# Model Database Migration

This document describes the migration from file-based model storage to database-based model storage.

## Overview

The model training and prediction system has been updated to store models in the database instead of individual files. This provides better versioning, centralized storage, and easier model management.

## Changes Made

### 1. Database Schema Updates

#### New Models Table
A new `models` table has been added to store model binaries:

```sql
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
```

#### Updated Results Tables
The existing results tables have been updated to include additional columns for directional metrics.

### 2. Updated Files

#### `utils/db/model_db_util.py`
- Added `Model` class for database model storage
- Added functions for saving/loading models from database
- Added versioning support with automatic version increment
- Added model activation/deactivation functionality

#### `tasks/ai/train_classifier.py`
- Updated to save models to database instead of files
- Removed file-based model storage
- Added version tracking in results

#### `tasks/ai/train_regression.py`
- Updated to save models to database instead of files
- Removed file-based model storage
- Added version tracking in results

#### `tasks/ai/predict.py`
- Updated to load models from database instead of files
- Added better error handling for missing models

## Setup Instructions

### 1. Create the Models Table

Run the SQL script to create the new models table:

```bash
# Option 1: Use the separate SQL file
psql -d your_database -f sql/create_models_table.sql

# Option 2: Use the updated create-tables.sql
psql -d your_database -f sql/create-tables.sql
```

### 2. Migrate Existing Models (Optional)

If you have existing `.joblib` model files, you can migrate them to the database:

```bash
python utils/migrate_models_to_db.py
```

### 3. Update Your Training Scripts

The training scripts now automatically save models to the database. No additional configuration is needed.

## Usage

### Training Models

Models are now automatically saved to the database when you run the training scripts:

```bash
# Train classifier models
python tasks/ai/train_classifier.py --source db

# Train regression models
python tasks/ai/train_regression.py --source db
```

### Making Predictions

The prediction script now loads models from the database:

```bash
python tasks/ai/predict.py
```

### Model Management

You can use the new utility functions to manage models:

```python
from utils.db.model_db_util import (
    save_model_to_db,
    load_model_from_db,
    get_latest_model_version,
    deactivate_model
)

# Save a model
success, version = save_model_to_db(model, "my_model", "event_name", "regression")

# Load the latest version of a model
model, version = load_model_from_db("my_model", "event_name", "regression")

# Load a specific version
model, version = load_model_from_db("my_model", "event_name", "regression", version=2)

# Deactivate a model version
deactivate_model("my_model", "event_name", "regression", version=1)
```

## Benefits

1. **Versioning**: Each model training run creates a new version
2. **Centralized Storage**: All models stored in one place
3. **Better Management**: Easy to activate/deactivate model versions
4. **Scalability**: No file system limitations
5. **Backup**: Models included in database backups

## Migration Notes

- Existing model files are not automatically deleted
- You can safely run the migration script multiple times
- The system will automatically use the latest active version of each model
- Old model files can be deleted after confirming successful migration

## Troubleshooting

### Model Not Found Errors
- Ensure the models table exists in your database
- Check that models have been trained and saved successfully
- Verify the model name, event, and type match exactly

### Version Conflicts
- The system automatically handles versioning
- Each training run creates a new version
- You can deactivate old versions if needed

### Performance Issues
- Large models may take longer to load from database
- Consider using model caching if needed
- Monitor database size as models are stored as binary data 