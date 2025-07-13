#!/bin/bash
# =====================================================
# NightScan Database Initialization Script
# =====================================================
# This script creates the NightScan database with the complete schema
# including all constraints, indexes, and initial data.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}NightScan Database Initialization${NC}"
echo "=================================="

# Check for required environment variables
if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ] || [ -z "$DB_NAME" ] || [ -z "$DB_USER" ]; then
    echo -e "${RED}Error: Required environment variables are not set${NC}"
    echo "Please set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
    echo ""
    echo "Example:"
    echo "  export DB_HOST=localhost"
    echo "  export DB_PORT=5432"
    echo "  export DB_NAME=nightscan"
    echo "  export DB_USER=nightscan_user"
    echo "  export DB_PASSWORD=your_password"
    exit 1
fi

# Database connection string
if [ -z "$DB_PASSWORD" ]; then
    PGPASSWORD=""
else
    export PGPASSWORD=$DB_PASSWORD
fi

echo -e "${YELLOW}Database Configuration:${NC}"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo ""

# Function to check if database exists
check_database_exists() {
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME
}

# Check if database exists
if check_database_exists; then
    echo -e "${YELLOW}Warning: Database '$DB_NAME' already exists${NC}"
    read -p "Do you want to DROP and RECREATE it? This will DELETE ALL DATA! (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Initialization cancelled."
        exit 0
    fi
    
    echo -e "${YELLOW}Dropping existing database...${NC}"
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "DROP DATABASE IF EXISTS $DB_NAME;"
fi

# Create database
echo -e "${GREEN}Creating database '$DB_NAME'...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME WITH ENCODING='UTF8';"

# Apply schema
echo -e "${GREEN}Creating database schema...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$(dirname "$0")/create_database.sql"

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
TABLES=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
TABLES=$(echo $TABLES | tr -d ' ')

if [ "$TABLES" -gt 0 ]; then
    echo -e "${GREEN}✓ Successfully created $TABLES tables${NC}"
    
    # Show created tables
    echo -e "\n${GREEN}Created tables:${NC}"
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "\dt"
    
    # Show initial data
    echo -e "\n${GREEN}Initial plan features:${NC}"
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT plan_type, plan_name, monthly_quota, price_monthly_cents FROM plan_features ORDER BY price_monthly_cents;"
    
    echo -e "\n${GREEN}Database initialization completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update your .env file with the database credentials"
    echo "2. Start the NightScan application"
else
    echo -e "${RED}Error: No tables were created${NC}"
    exit 1
fi