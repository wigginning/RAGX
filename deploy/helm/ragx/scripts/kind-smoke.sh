#!/usr/bin/env bash
# RAGX Helm chart — kind cluster smoke test (RX-INF-03 DoD).
#
# Builds the RAGX image, loads it into a kind cluster, installs the chart,
# waits for readiness, then curls /v1/health. Supports the "lite" (default)
# and "full" profiles.
#
#   bash scripts/kind-smoke.sh lite      # zero-middleware single process
#   bash scripts/kind-smoke.sh full      # ES + Neo4j + MinIO + PG + Redis + obs
#
# Prereqs: docker (daemon running), kind, kubectl, helm on PATH.
set -euo pipefail

PROFILE="${1:-lite}"          # lite | full
CLUSTER="${2:-ragx-smoke}"
CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$CHART_DIR/../.." && pwd)"
IMAGE_TAG="ragx:${PROFILE}"

case "$PROFILE" in
  lite) DOCKERFILE="$REPO_ROOT/Dockerfile"; VALUES_FILE="" ;;
  full) DOCKERFILE="$REPO_ROOT/Dockerfile.full"; VALUES_FILE="-f $CHART_DIR/values-lite.yaml" ;;
  *) echo "unknown profile: $PROFILE (use lite|full)" >&2; exit 1 ;;
esac

echo ">> profile=$PROFILE cluster=$CLUSTER image=$IMAGE_TAG"

if ! command -v kind >/dev/null 2>&1; then
  echo "kind not found — install from https://kind.sigs.k8s.io/docs/getting-started/" >&2
  exit 1
fi

# 1. create cluster (reuse if it already exists)
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo ">> creating kind cluster $CLUSTER"
  kind create cluster --name "$CLUSTER" --wait 180s
fi

# 2. build the RAGX image and load it into kind
echo ">> building $IMAGE_TAG from $DOCKERFILE"
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$REPO_ROOT"
kind load docker-image "$IMAGE_TAG" --name "$CLUSTER"

# 3. install the chart
echo ">> helm install ragx ($PROFILE)"
helm install ragx "$CHART_DIR" $VALUES_FILE \
  --namespace ragx --create-namespace --wait --timeout 600s

# 4. wait for the API pod and smoke /v1/health
kubectl -n ragx wait --for=condition=Ready pod -l app.kubernetes.io/instance=ragx --timeout=300s
kubectl -n ragx port-forward svc/ragx 18000:8000 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
sleep 4
if curl -fsS http://localhost:18000/v1/health; then
  echo "SMOKE_OK"
else
  echo "SMOKE_FAIL"
  kubectl -n ragx get pods
  exit 1
fi
