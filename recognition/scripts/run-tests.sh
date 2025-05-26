#!/bin/sh

rm -f /app-recognition/.coverage*
rm -f /app-recognition/coverage.xml
rm -rf /app-recognition/htmlcov/

find /app-recognition -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /app-recognition -name "*.pyc" -delete 2>/dev/null || true

find /app-recognition -name "config-*.py" -delete 2>/dev/null || true

set -e

pwd
ls -la

pip install coverage

cat > .coveragerc << 'EOF'
[run]
source = .
omit =
    */migrations/*
    */venv/*
    */virtualenv/*
    */__pycache__/*
    */staticfiles/*
    */media/*
    manage.py
    */settings/*
    */config*.py
    config*.py
    *config*.py
    db.sqlite3
    model_final.pth
    */test*
    */tests/*
    test_*.py
    *_test.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

[xml]
output = coverage.xml
EOF

coverage erase
coverage run --rcfile=.coveragerc manage.py test --verbosity=2

coverage xml -o /coverage/coverage.xml

sed -i 's#<source>/app-recognition#<source>recognition#' /coverage/coverage.xml
