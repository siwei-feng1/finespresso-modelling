SELECT 
    publisher,
    COUNT(CASE WHEN content IS NULL THEN 1 END) AS null_count,
    COUNT(CASE WHEN content IS NOT NULL THEN 1 END) AS not_null_count
FROM 
    news
GROUP BY 
    publisher
ORDER BY 
    publisher;