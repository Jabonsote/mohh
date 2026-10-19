"""Tests de las metricas de costo y de la conexion con el servidor."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mohh.llamea import metrics
from mohh.llamea.vllm import wait_for_server


def test_parse_prometheus_sums_labeled_series():
    text = (
        "# HELP vllm:request_success_total ok\n"
        'vllm:request_success_total{finished_reason="stop"} 3\n'
        'vllm:request_success_total{finished_reason="length"} 1\n'
        "vllm:num_requests_running 2\n"
        "not a number\n"
    )
    values = metrics.parse_prometheus(text)
    assert values["vllm:request_success_total"] == 4.0
    assert values["vllm:num_requests_running"] == 2.0


def test_counter_delta_and_cost_summary():
    before = {
        "vllm:request_inference_time_seconds_sum": 10.0,
        "vllm:request_prefill_time_seconds_sum": 4.0,
        "vllm:request_decode_time_seconds_sum": 6.0,
        "vllm:prompt_tokens_total": 100.0,
        "vllm:generation_tokens_total": 50.0,
        "vllm:request_success_total": 2.0,
        "vllm:time_to_first_token_seconds_sum": 1.0,
        "vllm:time_to_first_token_seconds_count": 2.0,
    }
    after = dict(before)
    after.update({
        "vllm:request_inference_time_seconds_sum": 30.0,   # +20 s de GPU
        "vllm:request_prefill_time_seconds_sum": 12.0,
        "vllm:request_decode_time_seconds_sum": 18.0,
        "vllm:prompt_tokens_total": 400.0,                  # +300
        "vllm:generation_tokens_total": 200.0,              # +150
        "vllm:request_success_total": 6.0,
        "vllm:time_to_first_token_seconds_sum": 5.0,
        "vllm:time_to_first_token_seconds_count": 6.0,
    })
    usage = {"requests": 4, "prompt_tokens": 300, "completion_tokens": 150,
             "total_tokens": 450, "inference_seconds": 40.0, "failed_requests": 1}
    summary = metrics.cost_summary(usage, before, after, gpu=None, wall_seconds=60.0)
    assert summary["source"] == "vllm"
    assert summary["gpu_seconds"] == 20.0
    assert summary["gpu_hours"] == pytest.approx(20.0 / 3600.0, abs=1e-6)
    assert summary["gpu_seconds_per_request"] == 5.0
    assert summary["gpu_tokens_per_second"] == 150.0 / 20.0
    assert summary["server_tokens_prompt"] == 300.0
    assert summary["tokens_per_second"] == 150 / 40.0
    assert summary["ttft_mean_seconds"] == round(4.0 / 4.0, 4)


def test_cost_summary_client_only_without_metrics():
    usage = {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5,
             "total_tokens": 15, "inference_seconds": 2.0, "failed_requests": 0}
    summary = metrics.cost_summary(usage, before=None, after=None)
    assert summary["source"] == "client-only"
    assert "gpu_seconds" not in summary
    assert summary["tokens_per_second"] == 2.5


def test_nvidia_smi_without_binary_returns_none(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(metrics.subprocess, "run", boom)
    assert metrics.nvidia_smi() is None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - API de http.server
        if self.path == "/metrics":
            body = b'vllm:num_requests_running 0\n'
        elif self.path == "/version":
            body = json.dumps({"version": "0.6.0"}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        elif self.path == "/models":
            body = b'{"data": []}'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silencio en los tests
        return


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_scrape_prometheus_and_server_version_against_stub():
    server = _serve()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}/v1"
    try:
        values = metrics.scrape_prometheus(base)
        assert values and values["vllm:num_requests_running"] == 0.0
        version = metrics.server_version(base)
        assert version == {"version": "0.6.0"}
        assert wait_for_server(base, timeout=5) is True
    finally:
        server.shutdown()


def test_scrape_and_wait_on_closed_port_return_none_or_false():
    # Puerto cerrado garantizado: se abre y se cierra para tomar uno libre.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    base = f"http://127.0.0.1:{port}/v1"
    assert metrics.scrape_prometheus(base, timeout=1) is None
    assert metrics.server_version(base, timeout=1) is None
    assert wait_for_server(base, timeout=1) is False
