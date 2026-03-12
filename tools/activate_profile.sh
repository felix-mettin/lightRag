#!/bin/zsh

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/activate_profile.sh <gb|iec|dlt>" >&2
  exit 1
fi

profile="$1"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

case "$profile" in
  gb|iec|dlt)
    ;;
  *)
    echo "Unsupported profile: $profile" >&2
    exit 1
    ;;
esac

config_src="$repo_root/config/profiles/config.${profile}.ini"
memory_profile="$profile"
memory_src="$repo_root/lightrag/config/profiles/annotation_memory.${memory_profile}.json"
config_dst="$repo_root/config.ini"
memory_dst="$repo_root/lightrag/config/annotation_memory.json"
env_file="$repo_root/.env"
working_dir="./data/rag_storage_${profile}"

if [[ ! -f "$config_src" ]]; then
  echo "Missing config profile: $config_src" >&2
  exit 1
fi

if [[ ! -f "$memory_src" ]]; then
  echo "Missing memory profile: $memory_src" >&2
  exit 1
fi

cp "$config_src" "$config_dst"
cp "$memory_src" "$memory_dst"

if grep -q '^WORKING_DIR=' "$env_file"; then
  sed -i.bak "s#^WORKING_DIR=.*#WORKING_DIR=${working_dir}#" "$env_file"
else
  printf '\nWORKING_DIR=%s\n' "$working_dir" >> "$env_file"
fi
rm -f "${env_file}.bak"

mkdir -p "$repo_root/data/rag_storage_${profile}"

echo "Activated profile: $profile"
echo "Config: $config_dst"
echo "Memory source profile: $memory_profile"
echo "Memory: $memory_dst"
echo "WORKING_DIR: $working_dir"
