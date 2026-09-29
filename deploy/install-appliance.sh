#!/bin/sh
# Install or refresh the execution appliance on this machine (idempotent).
# Run as the appliance user from the repo checkout at ~/projects/meridian:
#   deploy/install-appliance.sh
# Checks every prerequisite first and changes nothing unless all pass.
set -eu
repo="$(cd "$(dirname "$0")/.." && pwd)"
units="$HOME/.config/systemd/user"
missing=""

need() { missing="$missing\n  - $1"; }

[ "$repo" = "$HOME/projects/meridian" ] || need "repo must live at ~/projects/meridian (units use that path), found $repo"
for cmd in git uv nc curl Xvfb java; do
  command -v "$cmd" >/dev/null 2>&1 || need "command not found: $cmd"
done
[ -x "$HOME/ibc/gatewaystart.sh" ] || need "IBC not installed: ~/ibc/gatewaystart.sh (see deploy/meridian-gateway.service)"
if [ -f "$HOME/ibc/config.ini" ]; then
  [ "$(stat -c %a "$HOME/ibc/config.ini")" = "600" ] || need "~/ibc/config.ini holds the password; chmod 600 it"
else
  need "missing ~/ibc/config.ini (paper credentials, TradingMode=paper)"
fi
git -C "$repo" push --dry-run >/dev/null 2>&1 || need "git push from $repo fails (the cycle commits its journal)"

if [ -n "$missing" ]; then
  printf "not installed, fix first:%b\n" "$missing" >&2
  exit 1
fi

if [ ! -f "$HOME/.config/meridian/alert.env" ]; then
  echo "warning: no ~/.config/meridian/alert.env; failures will only reach the journal" >&2
fi

mkdir -p "$units"
cp "$repo"/deploy/meridian-xvfb.service "$repo"/deploy/meridian-gateway.service \
  "$repo"/deploy/meridian-paper-cycle.service "$repo"/deploy/meridian-paper-cycle.timer \
  "$repo"/deploy/meridian-alert@.service "$units/"
systemctl --user daemon-reload
systemctl --user enable --now meridian-xvfb.service meridian-gateway.service meridian-paper-cycle.timer
loginctl enable-linger "$USER"

systemctl --user --no-pager status meridian-xvfb meridian-gateway | grep -E '●|Active'
systemctl --user list-timers --no-pager | grep meridian
echo "installed. The gateway needs a minute or two to log in; then: nc -z 127.0.0.1 4002"
