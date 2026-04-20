# Database Schema: clandar-b9139.googleads_manger_calendar.ai_campaign_performance_mart

This view contains daily performance metrics for Google Ads campaigns.

## Columns:

- `date`: The date of the performance record (YYYY-MM-DD).
- `campaign_id`: The unique numerical identifier for the Google Ads campaign.
- `campaign_name`: The human-readable name of the campaign.
- `campaign_status`: The state of the campaign (e.g., ENABLED, PAUSED, REMOVED).
- `cost_inr`: Total spend for the day, formatted in Indian Rupees (INR).
- `impressions`: Number of times the ad was shown.
- `clicks`: Number of times the ad was clicked.
- `ctr`: Click-through rate (clicks / impressions).
- `conversions`: Total number of conversion events triggered.
- `cost_per_conversion`: Cost divided by conversions.
- `conversion_value`: The total monetary value tracked from the conversions.
- `roas`: Return on Ad Spend (conversion_value / cost_inr).
