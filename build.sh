#!/usr/bin/env bash
set -o errexit

pip install --prefer-binary -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
