#!/bin/sh

while ! curl -s "$SONAR_HOST_URL/api/system/status" | grep -q "UP"; do
  sleep 1
done
echo "SonarQube is ready."
