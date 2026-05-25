#!/usr/bin/env bash
# Pi-Seizure-Detection installer — sets up the MJPEG webcam streamer on a Pi 5.
# The companion Seizure-Processor service (on a separate Linux box) consumes
# this stream and does the actual seizure detection.
#
# First-time bootstrap on the Pi (run as the unprivileged service user, e.g. `aman`):
#
#     git clone https://github.com/daman20/Pi-Seizure-Detection.git ~/seizure-detection \
#       && bash ~/seizure-detection/deploy/install.sh
#
# Re-running is safe (idempotent). The systemd unit also pulls + uv-syncs on
# every restart, so updates need only:
#
#     sudo systemctl restart seizure-streamer
#
# Logs:    sudo journalctl -u seizure-streamer -f
# Status:  sudo systemctl status seizure-streamer
# Stream:  http://<this-pi-ip>:8080/stream.mjpg
# Preview: http://<this-pi-ip>:8080/

set -euo pipefail

REPO_URL="https://github.com/daman20/Pi-Seizure-Detection.git"
INSTALL_DIR="${HOME}/seizure-detection"
SERVICE_NAME="seizure-streamer"
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

log "uv sync (streamer deps only — small, no torch/mediapipe)"
uv sync

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
log "Stream URL:  http://${IP}:8080/stream.mjpg"
log "Preview:     http://${IP}:8080/"
log "Logs:        sudo journalctl -u ${SERVICE_NAME} -f"
log "Now configure the Linux processor box with SD_CAMERA_SOURCE=http://${IP}:8080/stream.mjpg"
