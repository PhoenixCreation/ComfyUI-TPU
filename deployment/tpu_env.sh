# Pinned TPU launch environment (implementation spec sections 5.2/7).
#
# Source this BEFORE the first python import of torch_xla in the process.
# The ComfyUI adapter also installs these values itself, but the deployment
# contract keeps the launch environment explicit and deterministic.
#
# docs/changes.md problems 1/4: without TPU_SKIP_MDS_QUERY the runtime blocks
# on the (unresolvable) GCE metadata server; the legacy multi-host variables
# must stay unset or PJRT takes a multi-host bootstrap path.

export PJRT_DEVICE=TPU
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v5litepod-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,4,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost

unset TPU_PROCESS_ADDRESSES 2>/dev/null || true
unset XRT_TPU_CONFIG 2>/dev/null || true

# Recommended (root): TPU runtime startup/shutdown is significantly faster
# with transparent hugepages:
#   echo always > /sys/kernel/mm/transparent_hugepage/enabled

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$(python3 -c 'import os,libtpu; print(os.path.dirname(libtpu.__file__))' 2>/dev/null)"