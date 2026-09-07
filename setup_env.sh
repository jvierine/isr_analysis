#!/bin/bash
# Create a virtual environment for the antistarlink ISR analysis pipeline.
# Run once on each machine. Activate before running: source ~/venv/antistarlink/bin/activate
set -e

VENV_DIR="${HOME}/venv/antistarlink"

python3 -m venv --system-site-packages "${VENV_DIR}"

"${VENV_DIR}/bin/pip" install --upgrade pip

# pyfftw needs libfftw3-dev (already installed on revontuli)
"${VENV_DIR}/bin/pip" install pyfftw

# digital_rf: GNU Radio / DigitalRF reader/writer
"${VENV_DIR}/bin/pip" install digital_rf

# MPI Python bindings
"${VENV_DIR}/bin/pip" install mpi4py

# Geographic coordinate conversion (required by fit_lpi)
"${VENV_DIR}/bin/pip" install git+https://github.com/jvierine/jcoord.git

echo ""
echo "Environment ready at ${VENV_DIR}"
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Run analysis:   mpirun -np 24 ${VENV_DIR}/bin/python3 run_analysis.py config/millstone_2023-09-05.json"
