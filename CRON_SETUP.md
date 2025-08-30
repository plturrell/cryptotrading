# Data Loading Automation Setup

## Overview
The cryptotrading platform includes both manual and automated data loading capabilities:
- **CLI Tool** (`cli_data_loader.py`) - Interactive command-line data loading
- **Scheduled Loader** (`scheduled_data_loader.py`) - Automated data loading for cron/scheduled tasks

## CLI Usage

### Basic Commands

```bash
# Check data source status
python3 cli_data_loader.py status

# Load crypto data (Yahoo Finance)
python3 cli_data_loader.py yahoo --symbols BTC ETH SOL --days 30 --interval 1d

# Load economic data (FRED)
python3 cli_data_loader.py fred --series DGS10 WALCL M2SL --days 365

# Load DEX data (GeckoTerminal)
python3 cli_data_loader.py gecko --networks ethereum polygon --pools 20

# Load all data sources
python3 cli_data_loader.py all --symbols BTC ETH --series DGS10 --networks ethereum

# Monitor active jobs
python3 cli_data_loader.py jobs --watch

# Cancel a job
python3 cli_data_loader.py cancel <job_id>
```

### Interactive Features
- Progress monitoring with visual progress bars
- Real-time job status updates
- Tabulated output for better readability

## Automated/Scheduled Loading

### Cron Setup

Add these entries to your crontab (`crontab -e`):

```bash
# Set environment variable for server location
DATA_LOADER_SERVER=http://localhost:5001

# Hourly crypto data updates (every hour)
0 * * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --crypto hourly --monitor

# Daily economic data updates (9 AM)
0 9 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --economic daily --monitor

# DEX data every 4 hours
0 */4 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --dex 4h --monitor

# Complete daily update at midnight
0 0 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --all daily --monitor --notify

# Weekly comprehensive update (Sunday 2 AM)
0 2 * * 0 /usr/bin/python3 /path/to/scheduled_data_loader.py --all weekly --monitor --notify
```

### Systemd Timer Setup (Alternative to Cron)

Create service file `/etc/systemd/system/crypto-data-loader.service`:

```ini
[Unit]
Description=Crypto Data Loader Service
After=network.target

[Service]
Type=oneshot
User=your-username
WorkingDirectory=/path/to/cryptotrading
ExecStart=/usr/bin/python3 /path/to/scheduled_data_loader.py --all daily --monitor
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create timer file `/etc/systemd/system/crypto-data-loader.timer`:

```ini
[Unit]
Description=Daily Crypto Data Loading
Requires=crypto-data-loader.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-data-loader.timer
sudo systemctl start crypto-data-loader.timer
```

## Schedule Configurations

### Crypto Data (Yahoo Finance)
- **Hourly**: 5 symbols, 1 day history, 15-minute intervals
- **Daily**: 10 symbols, 7 days history, 1-hour intervals  
- **Weekly**: 10 symbols, 30 days history, 1-day intervals

### Economic Data (FRED)
- **Daily**: 6 series, 30 days history
- **Weekly**: 8 series, 180 days history
- **Monthly**: 8 series, 365 days history

### DEX Data (GeckoTerminal)
- **4-hour**: 3 networks, 10 pools each
- **Daily**: 5 networks, 20 pools each
- **Weekly**: 7 networks, 50 pools each

## Monitoring

### Log Files
Logs are stored in `~/.cryptotrading/logs/`:
- `data_loader_YYYYMMDD.log` - Daily log files
- Includes timestamps, status updates, and error messages

### Check Status
```bash
# View today's log
tail -f ~/.cryptotrading/logs/data_loader_$(date +%Y%m%d).log

# Check cron execution
grep CRON /var/log/syslog | grep crypto

# Check systemd timer status
systemctl status crypto-data-loader.timer
systemctl list-timers
```

## Server Requirements

The Flask server must be running for data loading to work:

```bash
# Start server manually
python3 app.py

# Or run as a service (recommended for production)
# Create systemd service for the Flask app
```

## Environment Variables

Set these in your shell profile or cron environment:

```bash
export DATA_LOADER_SERVER=http://localhost:5001
export FRED_API_KEY=your_fred_api_key  # Optional, for FRED data
```

## Troubleshooting

### Common Issues

1. **Server not found**
   - Ensure Flask server is running: `python3 app.py`
   - Check server URL in environment variable

2. **Jobs timing out**
   - Increase timeout: `--timeout 600` (10 minutes)
   - Check network connectivity to data sources

3. **Cron not executing**
   - Check cron service: `systemctl status cron`
   - Verify paths are absolute in crontab
   - Check cron logs: `/var/log/cron.log`

4. **Permission denied**
   - Make scripts executable: `chmod +x *.py`
   - Check file ownership and permissions

## Data Flow

1. **Trigger**: Manual CLI or scheduled cron/timer
2. **API Call**: Sends request to Flask data loading service
3. **Job Creation**: Creates job in database with unique ID
4. **Async Processing**: Data fetched from sources (simulated)
5. **Progress Tracking**: Job progress updated in database
6. **Completion**: Job marked complete, data ready for use

## Production Recommendations

1. **Use a process manager** (systemd, supervisor) for Flask server
2. **Set up proper logging** with rotation
3. **Monitor disk space** for data storage
4. **Implement alerting** for failed jobs
5. **Use environment-specific configs** (dev/staging/prod)
6. **Consider using Celery** for production job queue management
7. **Add authentication** to data loading endpoints
8. **Set up backup** for loaded data

## API Endpoints

All endpoints are prefixed with `/api/odata/v4/DataLoadingService/`:

- `GET /getDataSourceStatus` - Check data source availability
- `GET /getActiveJobs` - List active loading jobs
- `POST /loadYahooFinanceData` - Load crypto data
- `POST /loadFREDData` - Load economic data
- `POST /loadGeckoTerminalData` - Load DEX data
- `POST /loadAllMarketData` - Load from all sources
- `POST /cancelLoadingJob` - Cancel a running job

## Next Steps

1. Configure API keys for real data sources
2. Implement actual data fetching (currently simulated)
3. Set up proper data storage (currently in-memory)
4. Add data validation and error handling
5. Implement retry logic for failed requests
6. Add data deduplication
7. Set up monitoring dashboard