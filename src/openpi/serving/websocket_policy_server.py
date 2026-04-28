import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=60,
            ping_timeout=120,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        # Route by websocket path: "/" -> single inference (legacy contract),
        # "/infer_batch" -> batched inference (list[obs] -> list[resp]). Any
        # other path is rejected with a policy-violation close so clients get
        # a clear signal instead of a silent hang.
        path = getattr(getattr(websocket, "request", None), "path", "/") or "/"
        logger.info(f"Connection from {websocket.remote_address} opened (path={path})")

        if path in ("", "/"):
            await self._single_handler(websocket)
        elif path == "/infer_batch":
            await self._batch_handler(websocket)
        else:
            logger.warning(f"Unknown websocket path '{path}', closing connection")
            await websocket.close(
                code=websockets.frames.CloseCode.POLICY_VIOLATION,
                reason=f"Unknown path: {path}",
            )

    async def _single_handler(self, websocket: _server.ServerConnection):
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        # Carry-over metrics recorded at the end of the previous iteration.
        # We log them on the *next* iteration so the single log line has the
        # full previous cycle: pack + send + wait for next request.
        prev_carry: dict | None = None
        while True:
            try:
                cycle_start = time.monotonic()

                t0 = time.monotonic()
                raw = await websocket.recv()
                wait_ms = (time.monotonic() - t0) * 1000

                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(raw)
                unpack_ms = (time.monotonic() - t0) * 1000
                unpack_bytes = len(raw) if isinstance(raw, (bytes, bytearray)) else -1

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                server_timing = {
                    "infer_ms": infer_time * 1000,
                    "wait_ms": wait_ms,
                    "unpack_ms": unpack_ms,
                    "unpack_bytes": unpack_bytes,
                }
                if prev_carry is not None:
                    server_timing.update(prev_carry)
                action["server_timing"] = dict(server_timing)

                _log_timing(logger, "/", action.get("policy_timing"), server_timing)

                t0 = time.monotonic()
                payload = packer.pack(action)
                pack_ms = (time.monotonic() - t0) * 1000

                t0 = time.monotonic()
                await websocket.send(payload)
                send_ms = (time.monotonic() - t0) * 1000

                prev_carry = {
                    "prev_pack_ms": pack_ms,
                    "prev_send_ms": send_ms,
                    "prev_total_ms": (time.monotonic() - cycle_start) * 1000,
                    "prev_send_bytes": len(payload) if isinstance(payload, (bytes, bytearray)) else -1,
                }

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _batch_handler(self, websocket: _server.ServerConnection):
        if not hasattr(self._policy, "infer_batch"):
            await websocket.close(
                code=websockets.frames.CloseCode.POLICY_VIOLATION,
                reason="Policy does not implement infer_batch.",
            )
            return

        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_carry: dict | None = None
        while True:
            try:
                cycle_start = time.monotonic()

                t0 = time.monotonic()
                raw = await websocket.recv()
                wait_ms = (time.monotonic() - t0) * 1000

                t0 = time.monotonic()
                obs_list = msgpack_numpy.unpackb(raw)
                unpack_ms = (time.monotonic() - t0) * 1000
                unpack_bytes = len(raw) if isinstance(raw, (bytes, bytearray)) else -1

                if not isinstance(obs_list, list):
                    raise TypeError(
                        f"/infer_batch expects a list of obs dicts, got {type(obs_list).__name__}"
                    )

                infer_time = time.monotonic()
                actions = self._policy.infer_batch(obs_list)
                infer_time = time.monotonic() - infer_time

                server_timing = {
                    "infer_ms": infer_time * 1000,
                    "batch_size": len(obs_list),
                    "wait_ms": wait_ms,
                    "unpack_ms": unpack_ms,
                    "unpack_bytes": unpack_bytes,
                }
                if prev_carry is not None:
                    server_timing.update(prev_carry)
                for action in actions:
                    action["server_timing"] = dict(server_timing)

                policy_timing = actions[0].get("policy_timing") if actions else None
                _log_timing(logger, "/infer_batch", policy_timing, server_timing)

                t0 = time.monotonic()
                payload = packer.pack(actions)
                pack_ms = (time.monotonic() - t0) * 1000

                t0 = time.monotonic()
                await websocket.send(payload)
                send_ms = (time.monotonic() - t0) * 1000

                prev_carry = {
                    "prev_pack_ms": pack_ms,
                    "prev_send_ms": send_ms,
                    "prev_total_ms": (time.monotonic() - cycle_start) * 1000,
                    "prev_send_bytes": len(payload) if isinstance(payload, (bytes, bytearray)) else -1,
                }

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _log_timing(
    log: logging.Logger,
    path: str,
    policy_timing: dict | None,
    server_timing: dict,
) -> None:
    """Emit a single-line per-request timing breakdown (all times in ms).

    Layout (one line, order preserved for easy grepping):

      path=<>  bs=<N>
      total=<policy total for this request>    srv=<infer wall time>
      model=<real GPU time>  in_tx=<input-transform>  stack=<batch stack>
      to_np=<device->host>   out_tx=<output-transform>
      wait=<recv idle>       unpack=<decode>          rx_kb=<request size>
      prev_pack=<last pack>  prev_send=<last send>    tx_kb=<last resp size>
      prev_total=<prev full cycle>

    Keeping ``wait`` separate from ``unpack`` matters: ``wait`` is
    ``recv()`` idle (client-bound), ``unpack`` is actual msgpack decode.
    """
    p = policy_timing or {}
    parts: list[str] = [f"path={path}", f"bs={p.get('batch_size', 1)}"]
    if "total_ms" in p:
        parts.append(f"total={p['total_ms']:.1f}")
    parts.append(f"srv={server_timing.get('infer_ms', 0.0):.1f}")
    if "model_ms" in p:
        parts.append(f"model={p['model_ms']:.1f}")
    if "input_transform_ms" in p:
        parts.append(f"in_tx={p['input_transform_ms']:.1f}")
    if "stack_ms" in p:
        parts.append(f"stack={p['stack_ms']:.1f}")
    if "to_numpy_ms" in p:
        parts.append(f"to_np={p['to_numpy_ms']:.1f}")
    if "output_transform_ms" in p:
        parts.append(f"out_tx={p['output_transform_ms']:.1f}")
    if "wait_ms" in server_timing:
        parts.append(f"wait={server_timing['wait_ms']:.1f}")
    if "unpack_ms" in server_timing:
        parts.append(f"unpack={server_timing['unpack_ms']:.1f}")
    rx_bytes = server_timing.get("unpack_bytes", -1)
    if rx_bytes >= 0:
        parts.append(f"rx_kb={rx_bytes / 1024:.1f}")
    if "prev_pack_ms" in server_timing:
        parts.append(f"prev_pack={server_timing['prev_pack_ms']:.1f}")
    if "prev_send_ms" in server_timing:
        parts.append(f"prev_send={server_timing['prev_send_ms']:.1f}")
    tx_bytes = server_timing.get("prev_send_bytes", -1)
    if tx_bytes >= 0:
        parts.append(f"tx_kb={tx_bytes / 1024:.1f}")
    if "prev_total_ms" in server_timing:
        parts.append(f"prev_total={server_timing['prev_total_ms']:.1f}")
    log.info("timing: " + " ".join(parts))


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
