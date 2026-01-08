# Rate Limiting Fix

## Problem
Yahoo Finance API was rate-limiting requests (429 errors) because:
- Too many rapid API calls
- No rate limiting between requests
- No retry logic with backoff
- Frontend making duplicate requests

## Solutions Implemented

### 1. Rate Limiting in Data Engine
- Added minimum 0.5 second delay between requests
- Exponential backoff on rate limit errors (2s, 4s)
- Automatic retry up to 2 times
- Falls back to cached data when rate limited

### 2. Better Caching
- 5-minute cache duration (300 seconds)
- Returns cached data even if expired when rate limited
- Cache persists across requests

### 3. Improved Error Handling
- Graceful degradation - shows symbols even if data unavailable
- Better error messages
- Frontend shows "Loading..." instead of crashing

### 4. Reduced API Calls
- Frontend now calls `view_watchlist` tool once instead of per-symbol
- Backend batches symbol lookups
- Added delays in batch operations

## How It Works Now

1. **First Request**: Fetches data from Yahoo Finance with rate limiting
2. **Cached Requests**: Returns cached data if available (< 5 min old)
3. **Rate Limited**: Waits and retries with exponential backoff
4. **Fallback**: Returns cached data even if expired, or shows symbol without price

## User Experience

- Symbols appear immediately in watchlist
- Prices load gradually (respecting rate limits)
- "Loading..." shown for unavailable data
- No crashes or errors in UI

## Configuration

You can adjust these values in `backend/data_engine.py`:
- `cache_duration = 300` - Cache time in seconds (5 minutes)
- `min_request_interval = 0.5` - Minimum seconds between requests
- `rate_limit_delay = 2.0` - Initial delay on rate limit (doubles on retry)

## Notes

- Yahoo Finance free API has strict rate limits
- For production, consider using a paid API or proxy
- Cache helps reduce API calls significantly
- Rate limiting is automatic and transparent to users

