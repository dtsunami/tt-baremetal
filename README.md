# tt — Tenstorrent Wormhole VSCode Tunnel on Koyeb

A Koyeb service that boots a [Tenstorrent Wormhole **n300s**](https://www.koyeb.com/docs/hardware/tenstorrent-n300)
instance and starts a **VSCode remote tunnel**, so you can attach your local VSCode
(or vscode.dev in a browser) directly to the Wormhole box and develop against
`tt-metal` / `tt-metalium`.

It is built from the official
[`koyeb/tenstorrent-examples`](https://github.com/koyeb/tenstorrent-examples) `tt-vsc-tunnel`
setup. Koyeb auto-builds the [`Dockerfile`](./Dockerfile) from this repo on every push to `main`.

## What's in the image

- **Base:** `ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc`
- Docker-in-Docker (so you can build/run containers on the instance)
- The VSCode CLI, launched as a named tunnel via Koyeb's entrypoint

The default tunnel name is `tt-on-koyeb` — override it with the `VSC_NODE_NAME`
environment variable on the service.

## Deploy on Koyeb

> Requires a Koyeb account with access to the Tenstorrent private preview.

### Option A — Koyeb Control Panel

1. **Create Service → GitHub**, and select this repo (`dtsunami/tt`), branch `main`.
2. Koyeb auto-detects the `Dockerfile` and builds it.
3. Configure the service with these settings:

   | Setting          | Value                                   |
   | ---------------- | --------------------------------------- |
   | Service type     | **Worker** (no public HTTP port needed) |
   | Instance type    | `gpu-tenstorrent-n300s`                 |
   | Region           | `na` (Washington, D.C.)                 |
   | Privileged       | **Enabled** (required for DinD + device access) |
   | Volume           | 10 GB mounted at `/workdir`             |
   | Scaling          | Min/Max 1                               |

4. Deploy.

### Option B — Koyeb CLI

```bash
koyeb deploy . tt/vsc-tunnel \
  --instance-type gpu-tenstorrent-n300s \
  --region na \
  --type worker \
  --privileged \
  --volume tt-workdir:/workdir \
  --env VSC_NODE_NAME=tt-on-koyeb
```

> The `tt-workdir` volume must exist (`koyeb volume create tt-workdir --size 10 --region na`).
> Persisting `/workdir` keeps your GitHub tunnel auth across redeploys.

## Connect

1. After the service starts, open its **Logs** in Koyeb. You'll see a one-time
   GitHub device-auth prompt:

   ```
   To grant access to the server, please log into https://github.com/login/device
   and use code XXXX-XXXX
   ```

   Open that URL, enter the code, and authorize.
2. In your **local VSCode**: `Ctrl/Cmd+Shift+P` → **Remote-Tunnels: Connect to Tunnel…**
   → sign in with the same GitHub account → pick the device (`tt-on-koyeb`).
3. You're now editing on the Wormhole instance. Open a terminal and verify the
   hardware:

   ```bash
   tt-smi
   ```

> The device-auth step repeats after each redeploy **unless** the `/workdir`
> volume persists the tunnel credentials (it does, in the config above).

## Notes

- This is a **worker** service (no inbound HTTP), so it has no public URL and no
  health-check port — the tunnel is outbound to Microsoft's tunnel relay.
- The base image is pinned to a concrete tag (`...:v0.71.2`) via the `BASE_IMAGE`
  build arg, so Koyeb's build cache key stays stable across rebuilds. Bump it
  deliberately to move to a newer `tt-metalium` — see the
  [available tags](https://github.com/tenstorrent/tt-metal/pkgs/container/tt-metal%2Ftt-metalium-ubuntu-22.04-release-amd64).
