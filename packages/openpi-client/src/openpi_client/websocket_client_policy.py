import logging
import time
from typing import Dict, List, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    # Subclasses override to reach a different server path (e.g. /infer_batch).
    _PATH: str = ""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        if self._PATH:
            self._uri = self._uri.rstrip("/") + self._PATH
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass


class WebsocketBatchClientPolicy(WebsocketClientPolicy):
    """Batched counterpart to :class:`WebsocketClientPolicy`.

    Connects to the server's ``/infer_batch`` path and exchanges a msgpack
    list-of-obs for a msgpack list-of-responses in a single round trip.
    Instantiate alongside (not instead of) :class:`WebsocketClientPolicy` when
    a client wants to mix single- and batch-inference calls against the same
    server process.
    """

    _PATH: str = "/infer_batch"

    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:  # noqa: UP006
        data = self._packer.pack(list(obs_list))
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        out = msgpack_numpy.unpackb(response)
        if not isinstance(out, list):
            raise RuntimeError(f"/infer_batch expected list response, got {type(out).__name__}")
        return out

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # The batch endpoint has no single-infer contract; callers should use
        # a regular WebsocketClientPolicy for that. Provide a thin shim so the
        # BasePolicy abstract surface stays satisfied.
        return self.infer_batch([obs])[0]
