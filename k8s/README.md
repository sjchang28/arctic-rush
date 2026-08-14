# arctic-rush on local Kubernetes

## Setup

```
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl get pvc -n arctic-rush   # wait for Bound
```

Load the built image into the local cluster (skip if Docker Desktop k8s — it
shares the host Docker daemon):

```
minikube image load arctic-rush-train      # minikube
k3d image import arctic-rush-train         # k3s / k3d
```

Launch a run:

```
RUN_ID=baseline envsubst '$RUN_ID' < k8s/job-template.yaml | kubectl apply -f -
```

## GPU sharing on one physical GPU

Stock Kubernetes + the NVIDIA device plugin advertises `nvidia.com/gpu` as
**whole, non-fractional units**. On a box with one physical GPU, that means
only **one** Job's pod will ever reach `Running` at a time by default — every
other Job you `kubectl apply` sits `Pending` with a
`0/1 nvidia.com/gpu` scheduling event (`kubectl describe pod <name>` will
show it) until the first Job completes and releases the GPU.

To get true concurrent scheduling on one card, enable **NVIDIA GPU
time-slicing** in the device plugin. This is the recommended path for this
project over NVIDIA MPS — MPS setup on Docker Desktop/WSL2 GPU passthrough is
materially more fragile to get working locally, while time-slicing is a
supported `ConfigMap` toggle on Docker Desktop k8s, minikube
(`--driver=docker` + nvidia-container-toolkit), and k3s alike.

Apply a time-slicing config to the device plugin, e.g.:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
  namespace: kube-system
data:
  time-slicing: |
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 2
```

Then point the device plugin DaemonSet at this ConfigMap (via its Helm
`config.name` value or `--config-file` flag, depending on how it was
installed) and restart it.

Keep `replicas` modest (2-3). This model already risks VRAM pressure at full
`NUM_UNROLL_STEPS` (see `src/model/config.py`), and time-slicing gives no memory
isolation between concurrent runs on the same physical GPU — an OOM in one
run can affect others sharing the slice.

Verify the advertised count increased before assuming concurrent Jobs will
schedule:

```
kubectl describe node | grep nvidia.com/gpu
# or
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'
```

## Checking results

```
kubectl get pods -n arctic-rush -w
kubectl logs -n arctic-rush job/train-baseline -f
```

Checkpoints and logs land on the shared PVC under `/data/models/<RUN_ID>/`
and `/data/logs/<RUN_ID>/` (including a `tensorboard/` subdir per run) —
separate trees per run, safe to run concurrently. Point TensorBoard at the
mounted PVC (via `kubectl cp` or a throwaway debug pod mounting the same
PVC) to compare runs:

```
tensorboard --logdir /path/to/mounted/data/logs
```
