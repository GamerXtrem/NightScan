#!/bin/bash

# Test script for NightScan API Gateway

set -e

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
TEST_USER="${TEST_USER:-testuser}"
TEST_PASS="${TEST_PASS:-testpass123}"

echo "Testing NightScan API Gateway at $GATEWAY_URL"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local token=$4
    local expected_status=$5
    
    echo -n "Testing $method $endpoint... "
    
    if [ -n "$token" ]; then
        auth_header="-H \"Authorization: Bearer $token\""
    else
        auth_header=""
    fi
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X $method "$GATEWAY_URL$endpoint" \
            -H "Content-Type: application/json" \
            $auth_header \
            -d "$data" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X $method "$GATEWAY_URL$endpoint" \
            $auth_header 2>/dev/null)
    fi
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}✓${NC} ($status_code)"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 0
    else
        echo -e "${RED}✗${NC} (Expected: $expected_status, Got: $status_code)"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 1
    fi
}

# Test 1: Health check (no auth required)
echo -e "\n1. Testing health endpoint (no auth):"
test_endpoint "GET" "/health" "" "" "200"

# Test 2: Try protected endpoint without auth (should fail)
echo -e "\n2. Testing protected endpoint without auth:"
test_endpoint "GET" "/api/v1/detections" "" "" "401"

# Test 3: Login
echo -e "\n3. Testing login:"
login_response=$(curl -s -X POST "$GATEWAY_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$TEST_USER\",\"password\":\"$TEST_PASS\"}")

if echo "$login_response" | jq -e '.success' > /dev/null 2>&1; then
    ACCESS_TOKEN=$(echo "$login_response" | jq -r '.data.access_token')
    REFRESH_TOKEN=$(echo "$login_response" | jq -r '.data.refresh_token')
    echo -e "${GREEN}✓${NC} Login successful"
    echo "Access token: ${ACCESS_TOKEN:0:20}..."
else
    echo -e "${RED}✗${NC} Login failed"
    echo "$login_response" | jq '.'
    echo "Please create test user first:"
    echo "curl -X POST $GATEWAY_URL/api/auth/register -H 'Content-Type: application/json' -d '{\"username\":\"$TEST_USER\",\"password\":\"$TEST_PASS\",\"email\":\"test@example.com\"}'"
    exit 1
fi

# Test 4: Access protected endpoint with token
echo -e "\n4. Testing protected endpoint with JWT:"
test_endpoint "GET" "/api/v1/detections?page=1" "" "$ACCESS_TOKEN" "200"

# Test 5: Test token verification
echo -e "\n5. Testing token verification:"
test_endpoint "GET" "/api/auth/verify" "" "$ACCESS_TOKEN" "200"

# Test 6: Test refresh token
echo -e "\n6. Testing token refresh:"
test_endpoint "POST" "/api/auth/refresh" "{\"refresh_token\":\"$REFRESH_TOKEN\"}" "" "200"

# Test 7: Test rate limiting
echo -e "\n7. Testing rate limiting (may take a while):"
echo -n "Sending multiple requests... "
for i in {1..150}; do
    status=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$GATEWAY_URL/health")
    if [ "$status" = "429" ]; then
        echo -e "${GREEN}✓${NC} Rate limit working (triggered after $i requests)"
        break
    fi
done

# Test 8: Test CORS headers
echo -e "\n8. Testing CORS headers:"
cors_response=$(curl -s -I -X OPTIONS "$GATEWAY_URL/api/v1/detections" \
    -H "Origin: http://localhost:3000" \
    -H "Access-Control-Request-Method: GET")

if echo "$cors_response" | grep -q "Access-Control-Allow-Origin"; then
    echo -e "${GREEN}✓${NC} CORS headers present"
else
    echo -e "${RED}✗${NC} CORS headers missing"
fi

# Test 9: Test analytics endpoint
echo -e "\n9. Testing analytics endpoint:"
test_endpoint "GET" "/api/analytics/metrics" "" "$ACCESS_TOKEN" "200"

# Test 10: Test logout
echo -e "\n10. Testing logout:"
test_endpoint "POST" "/api/auth/logout" "" "$ACCESS_TOKEN" "200"

echo -e "\n=========================================="
echo "Gateway tests completed!"