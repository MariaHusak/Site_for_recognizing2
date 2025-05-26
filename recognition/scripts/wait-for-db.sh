#!/bin/sh

while ! pg_isready -h db -p 5432 -q; do
  sleep 1
done
echo "Database is ready."
