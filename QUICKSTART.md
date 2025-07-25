# NightScan Quick Start Guide

This guide will help you get NightScan running locally for development.

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis (optional, for caching)

## Quick Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements-macos.txt  # or requirements.txt for Linux
```

### 2. Set Up Database

Option A: Quick development setup (recommended for first-time setup)
```bash
# This script will create a PostgreSQL user and database
./setup_dev_db.sh
```

Option B: Full schema initialization
```bash
# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=nightscan
export DB_USER=nightscan
export DB_PASSWORD=nightscan_secure_password_2025

# Initialize database with full schema
cd database
./init_database.sh
cd ..
```

### 3. Create Test User

```bash
# This will create a test user for login
python create_test_user_fixed.py

# Login credentials:
# Username: testuser
# Password: testpass123
```

### 4. Start the Application

```bash
# Start with development configuration
./start_dev.sh
```

The application will be available at http://localhost:8000

## Common Issues and Solutions

### Issue: CSRF Token Missing
- **Cause**: CSRF protection was disabled
- **Status**: Fixed - CSRF protection is now enabled

### Issue: Worker Timeout
- **Cause**: Database connection issues during startup
- **Solution**: The start_dev.sh script uses longer timeouts for development

### Issue: Database Connection Failed
- **Solution**: 
  1. Ensure PostgreSQL is running: `pg_isready`
  2. Run `./setup_dev_db.sh` to create the database
  3. Check your connection string in environment variables

### Issue: Login Not Working
- **Solution**: 
  1. Ensure you've created a test user: `python create_test_user_fixed.py`
  2. Check that the database tables were created
  3. Look for errors in the console output

## Development Tips

1. **Environment Variables**: The application uses these key variables:
   - `SECRET_KEY`: Auto-generated if not set
   - `SQLALCHEMY_DATABASE_URI`: PostgreSQL connection string
   - `FLASK_ENV`: Set to 'development' for debug mode
   - `WORKER_TIMEOUT`: Gunicorn worker timeout (default: 120s)

2. **Database**: In development mode, the app will attempt to create tables automatically if they don't exist.

3. **Logs**: Check the console output for detailed error messages and debugging information.

4. **Debug Mode**: The start_dev.sh script runs in debug mode with auto-reload enabled.

## Next Steps

1. Access the web interface at http://localhost:8000
2. Login with the test user credentials
3. Upload audio files for wildlife detection
4. Check the dashboard for real-time updates

For production deployment, see the main README.md file.