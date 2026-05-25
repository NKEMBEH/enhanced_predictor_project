#!/usr/bin/env bash
set -o errexit

# Step 1: Install heavy scientific packages that have cp314 binary wheels
pip install --only-binary :all: \
  "numpy>=2.0.0" \
  "scipy>=1.16.0"

# Step 2: Install scikit-learn build tools into the venv
pip install --prefer-binary \
  "Cython>=3.0.10,<3.2.0" \
  "meson-python>=0.16.0" \
  "pybind11"

# Step 3: Install scikit-learn without an isolated build env.
# This makes it use the scipy 1.16.0 already installed above instead of
# trying to compile scipy 1.15.x from source (which needs gfortran, not
# available on Render free tier).
pip install --no-build-isolation --prefer-binary "scikit-learn>=1.6.0,<2.0"

# Step 4: Install all remaining packages (already-installed ones are skipped)
pip install --prefer-binary -r requirements.txt

python manage.py collectstatic --noinput
python manage.py migrate
