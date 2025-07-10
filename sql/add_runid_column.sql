-- Add runid column to price_moves table
ALTER TABLE price_moves ADD COLUMN runid BIGINT NULL;

-- Add index on runid for better query performance
CREATE INDEX idx_price_moves_runid ON price_moves(runid);

-- Add comment to document the column
COMMENT ON COLUMN price_moves.runid IS 'Run ID to separate different price move calculation runs'; 