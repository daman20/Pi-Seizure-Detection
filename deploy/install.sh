#!/usr/bin/env bash
# Pi-Seizure-Detection auto-deploy installer.
#
# First-time bootstrap (run on the Pi as the unprivileged service user, e.g. `aman`):
#
#     git clone https://github.com/daman20/Pi-Seizure-Detection.git ~/seizure-detection \
#       && bash ~/seizure-detection/deploy/install.sh
#
# Re-running the script is safe (idempotent). It pulls the latest commit,
# refreshes deps, reinstalls the systemd unit if changed, and restarts the
# service. The systemd unit itself also pulls + syncs on every restart, so
# normal updates only require:
#
#     sudo systemctl restart seizure-detection
#
# Logs:    sudo journalctl -u seizure-detection -f
# Status:  sudo systemctl status seizure-detection
# Dashboard reachable on the LAN at http://<pi-ip>:8000/

set -euo pipefail

REPO_URL="https://github.com/daman20/Pi-Seizure-Detection.git"
INSTALL_DIR="${HOME}/seizure-detection"
SERVICE_NAME="seizure-detection"
SERVICE_USER="$(id -un)"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

if [ "${SERVICE_USER}" = "root" ]; then
    echo "Refusing to install under root. Re-run as the service user (e.g. aman)." >&2
    exit 1
fi

log "System prerequisites"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libglib2.0-0 \
    v4l-utils

log "Ensuring ${SERVICE_USER} can access /dev/video* (video group)"
sudo usermod -aG video "${SERVICE_USER}"

log "Installing uv (if not present)"
if ! command -v uv &>/dev/null && [ ! -x "${HOME}/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

log "Cloning / pulling repo into ${INSTALL_DIR}"
if [ -d "${INSTALL_DIR}/.git" ]; then
    git -C "${INSTALL_DIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"

log "uv sync (this may take a few minutes on first run — torch + mediapipe)"
uv sync

if [ ! -f "models/face_landmarker.task" ]; then
    log "Downloading MediaPipe face landmarker"
    uv run python scripts/download_face_model.py
else
    log "Face landmarker model already present"
fi

# Uncomment to export YOLO to NCNN for ~2-4x CPU speedup on Pi 5. Adds ~5 min
# to first install and produces models/yolo11n-pose_ncnn_model/.
# log "Exporting YOLO11n-pose to NCNN"
# uv run python scripts/export_ncnn.py

log "Installing systemd unit"
TMP_UNIT="$(mktemp)"
sed \
    -e "s|__USER__|${SERVICE_USER}|g" \
    -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
    "${INSTALL_DIR}/deploy/${SERVICE_NAME}.service.template" > "${TMP_UNIT}"
sudo install -m 0644 "${TMP_UNIT}" "/etc/systemd/system/${SERVICE_NAME}.service"
rm -f "${TMP_UNIT}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

log "(Re)starting service"
sudo systemctl restart "${SERVICE_NAME}"

sleep 3
log "Status:"
sudo systemctl --no-pager status "${SERVICE_NAME}" | head -15 || true

IP="$(hostname -I | awk '{print $1}')"
log "Done."
log "Dashboard: http://${IP}:8000/"
log "Logs:      sudo journalctl -u ${SERVICE_NAME} -f"
