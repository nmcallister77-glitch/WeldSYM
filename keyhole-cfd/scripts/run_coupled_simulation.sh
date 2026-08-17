#!/usr/bin/env bash
# run_coupled_simulation.sh — Production launcher for keyhole CFD + thermo-mechanical FEA
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${CASE:-${ROOT}/openfoam}"
NP_CFD="${NP_CFD:-32}"
NP_FEA="${NP_FEA:-8}"
MATERIAL="${MATERIAL:-Ti6Al4V}"
COUPLING="${COUPLING:-one_way}"   # one_way | fully_coupled

log() { echo "[$(date -Iseconds)] $*"; }

# --- Environment ---
export WM_PROJECT_DIR="${WM_PROJECT_DIR:-/opt/openfoam2312}"
source "${WM_PROJECT_DIR}/etc/bashrc" 2>/dev/null || true

log "Configuring case (material=${MATERIAL})"
python3 "${ROOT}/scripts/configure_case.py" \
  --config "${ROOT}/config/simulation_master.yaml"

cd "${CASE}"

# --- Mesh ---
if [[ ! -d "constant/polyMesh" ]]; then
  log "Generating mesh: blockMesh + snappyHexMesh"
  blockMesh
  snappyHexMesh -overwrite
  checkMesh -allTopology -allGeometry
fi

# --- Decompose ---
if [[ ! -d "processor0" ]]; then
  log "Decomposing for ${NP_CFD} ranks"
  decomposePar -force
fi

# --- Metric watcher ---
python3 "${ROOT}/scripts/extract_metrics.py" \
  "${CASE}/postProcessing" \
  --watch --interval 5 &
METRICS_PID=$!
trap 'kill ${METRICS_PID} 2>/dev/null || true' EXIT

run_cfd() {
  log "Starting laserKeyholeVoF (${NP_CFD} MPI ranks)"
  mpirun -np "${NP_CFD}" laserKeyholeVoF -parallel
}

run_fea_oneway() {
  log "One-way FEA: map CFD temperature -> CalculiX"
  # foamToEnsight or custom mapFieldsToCalculiX
  mapFieldsToCalculiX -sourceTime latest
  ccx "${ROOT}/fea/calculix_thermomech"
}

run_fea_coupled() {
  log "Fully-coupled via preCICE"
  export PRECICE_CONFIG="${ROOT}/scripts/preCICE_config.xml"
  mpirun -np "${NP_CFD}" laserKeyholeVoF -parallel &
  CFD_PID=$!
  mpirun -np "${NP_FEA}" ccx_preCICE "${ROOT}/fea/calculix_thermomech" &
  FEA_PID=$!
  wait "${CFD_PID}" "${FEA_PID}"
}

case "${COUPLING}" in
  one_way)
    run_cfd
    run_fea_oneway
    ;;
  fully_coupled)
    run_fea_coupled
    ;;
  *)
    echo "Unknown COUPLING=${COUPLING}" >&2
    exit 1
    ;;
esac

# --- Reconstruct & export ---
reconstructPar -latestTime
python3 "${ROOT}/scripts/export_vtk_xdmf.py" "${CASE}/postProcessing"
python3 "${ROOT}/scripts/extract_metrics.py" "${CASE}/postProcessing" \
  --fea "${ROOT}/fea/calculix_thermomech.frd" \
  --output "${ROOT}/postProcessing/metrics/final_metrics.json"

log "Simulation complete. VTK/Xdmf in postProcessing/export/"
