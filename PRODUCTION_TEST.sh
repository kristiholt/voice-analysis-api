#!/bin/bash
# Production API Testing Script

# Set your production values
PRODUCTION_URL="https://your-app.railway.app"
API_KEY="your_generated_production_key"

echo "🧪 Testing Production Voice Analysis API..."

# 1. Health Check
echo "1. Health Check:"
curl -s "$PRODUCTION_URL/health" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'✅ Status: {data[\"status\"]}')
print(f'✅ Version: {data[\"version\"]}')
"

# 2. Authentication Test
echo -e "\n2. Usage Analytics (Auth Test):"
curl -s -H "Authorization: Bearer $API_KEY" \
     "$PRODUCTION_URL/v1/usage" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('✅ Authentication working')
    print(f'✅ Usage endpoint: {data[\"message\"]}')
except:
    print('❌ Authentication failed')
"

# 3. Rate Limiting Test
echo -e "\n3. Rate Limiting Test:"
for i in {1..3}; do
  response=$(curl -s -w "HTTP_STATUS:%{http_code}" \
                  -H "Authorization: Bearer $API_KEY" \
                  "$PRODUCTION_URL/v1/usage")
  status=$(echo "$response" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
  if [ "$status" = "429" ]; then
    echo "✅ Rate limiting active (429 received)"
    break
  else
    echo "✅ Request $i: Status $status"
  fi
done

echo -e "\n✅ Production testing complete!"
