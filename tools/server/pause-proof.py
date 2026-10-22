#!/usr/bin/env python3
"""Bounded native pause proof. Uses synthetic text; never calls TTS."""
import argparse
import hashlib
import http.client
import json
import pathlib
import socket
import threading
import time

PROMPT = ("<|im_start|>system\nYou explain technical topics clearly.<|im_end|>\n"
          "<|im_start|>user\nExplain how a sailboat moves against the wind. "
          "Give a detailed explanation in several paragraphs.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
STATE_KEYS = ("id_task", "generated", "prompt_tokens", "target_pos", "draft_pos")

def request(port, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    started = time.perf_counter_ns()
    try:
        connection.request("GET" if body is None else "POST", path,
                           None if body is None else json.dumps(body),
                           {} if body is None else {"Content-Type": "application/json"})
        response = connection.getresponse()
        value = json.loads(response.read())
        assert response.status == 200, (response.status, value)
        return value, (time.perf_counter_ns() - started) / 1e6
    finally:
        connection.close()

def control(port, completion_id, action):
    value, elapsed = request(port, "/v1/chat/completions/control",
                             {"id": completion_id, "action": action})
    assert value.get("success"), value
    return {"receipt": value["pause"], "roundtrip_ms": elapsed}

def seed(port):
    value, _ = request(port, "/completion", {"prompt": PROMPT, "n_predict": 0,
                       "cache_prompt": True, "id_slot": 0, "temperature": 0, "seed": 42})
    assert value.get("tokens_predicted") == 0, value
    return value.get("timings")

class Stream:
    def __init__(self, port, tokens=256, id_slot=0):
        self.port = port
        self.limit = tokens
        self.id_slot = id_slot
        self.cv = threading.Condition()
        self.connection = None
        self.id = None
        self.events = []
        self.tokens = []
        self.text = ""
        self.final = None
        self.error = None
        self.done = False
        self.aborted = False
        self.started_ns = time.perf_counter_ns()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            self.connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=40)
            payload = {"prompt": PROMPT, "max_tokens": self.limit, "stream": True,
                       "temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True,
                       "cache_prompt": True, "return_tokens": True, "verbose": True}
            if self.id_slot is not None:
                payload["id_slot"] = self.id_slot
            self.connection.request("POST", "/v1/completions", json.dumps(payload),
                                    {"Content-Type": "application/json"})
            response = self.connection.getresponse()
            assert response.status == 200, (response.status, response.read())
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                row = json.loads(data)
                assert "error" not in row, row
                now = time.perf_counter_ns()
                with self.cv:
                    self.id = row.get("id", self.id)
                    for choice in row.get("choices", []):
                        self.text += choice.get("text", "")
                        if choice.get("finish_reason"):
                            self.final = row
                    detail = row.get("__verbose", {})
                    if self.final is None and detail.get("tokens"):
                        self.tokens.extend(detail["tokens"])
                        self.events.append({"ns": now, "generated": detail["tokens_predicted"],
                                            "tokens": detail["tokens"]})
                    self.cv.notify_all()
            if not self.aborted:
                assert self.final is not None, "stream ended without completion"
                assert len(self.tokens) == self.limit, (len(self.tokens), self.final)
        except BaseException as error:
            if not self.aborted:
                self.error = repr(error)
        finally:
            if self.connection:
                self.connection.close()
            with self.cv:
                self.done = True
                self.cv.notify_all()

    def wait(self, predicate, timeout=40):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not predicate():
                assert not self.error, self.error
                assert not self.done, "generation ended before checkpoint"
                remaining = deadline - time.monotonic()
                assert remaining > 0, "stream checkpoint timeout"
                self.cv.wait(min(remaining, 0.1))

    def close(self):
        self.aborted = True
        if self.connection and self.connection.sock:
            try:
                self.connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "reader did not stop"

    def result(self):
        self.thread.join(timeout=40)
        assert not self.thread.is_alive() and not self.error, self.error
        return {"completion_id": self.id, "tokens": self.tokens,
                "text_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                "text": self.text, "first_token_ms": (self.events[0]["ns"] - self.started_ns)/1e6,
                "elapsed_ms": (time.perf_counter_ns()-self.started_ns)/1e6,
                "timings": self.final.get("timings"),
                "usage": self.final.get("usage")}

def sample(port, pause_points):
    seed_timing = seed(port)
    stream = Stream(port)
    pauses = []
    paused = False
    try:
        for threshold in pause_points:
            stream.wait(lambda: len(stream.tokens) >= threshold)
            p = control(port, stream.id, "pause")
            paused = True
            time.sleep(0.2)
            held = control(port, stream.id, "pause")
            assert {k:p["receipt"][k] for k in STATE_KEYS} == {
                k:held["receipt"][k] for k in STATE_KEYS}, (p, held)
            resumed_send_ns = time.perf_counter_ns()
            resumed = control(port, stream.id, "resume")
            paused = False
            assert {k:p["receipt"][k] for k in STATE_KEYS} == {
                k:resumed["receipt"][k] for k in STATE_KEYS}, (p, resumed)
            generated = p["receipt"]["generated"]
            stream.wait(lambda: any(e["generated"] > generated for e in stream.events))
            first = next(e for e in stream.events if e["generated"] > generated)
            assert first["ns"] >= resumed_send_ns, (first, resumed_send_ns)
            pauses.append({"after_tokens": threshold, "pause": p, "held": held,
                           "resume": resumed, "resume_to_api_token_ms":
                           (first["ns"]-resumed_send_ns)/1e6})
        result = stream.result()
        result.update(pauses=pauses, seed_timings=seed_timing)
        return result
    finally:
        if paused and stream.id:
            control(port, stream.id, "resume")
        if not stream.done:
            stream.close()

def _snapshot(stream):
    with stream.cv:
        last_generated = stream.events[-1]["generated"] if stream.events else 0
        return len(stream.tokens), last_generated


def _disconnect_control(port, completion_id):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps({"id": completion_id, "action": "pause"}).encode()
    try:
        connection.connect()
        connection.putrequest("POST", "/v1/chat/completions/control")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        connection.send(body)
    finally:
        if connection.sock:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()


def two_slot_probe(port):
    seed(port)
    first = Stream(port, id_slot=0)
    second = Stream(port, id_slot=1)
    queued = None
    paused = False
    try:
        first.wait(lambda: len(first.tokens) >= 12)
        second.wait(lambda: len(second.tokens) >= 12)
        before = {"first": _snapshot(first), "second": _snapshot(second)}
        pause = control(port, first.id, "pause")
        paused = True
        client_at_receipt = {"first": _snapshot(first), "second": _snapshot(second)}
        def native_snapshot():
            slots, _ = request(port, "/slots")
            return {s["id"]: {"task":s["id_task"], "prompt":s["n_prompt_tokens"],
                    "decoded":s["next_token"][0]["n_decoded"]} for s in slots}
        native_at_receipt = native_snapshot()
        # A pause ACK and SSE data travel on separate connections. Drain tokens
        # already counted by the native slot before checking the client remains still.
        first.wait(lambda: _snapshot(first)[1] >= native_at_receipt[0]["decoded"])
        second.wait(lambda: _snapshot(second)[1] >= native_at_receipt[1]["decoded"])
        at_receipt = {"first": _snapshot(first), "second": _snapshot(second)}
        time.sleep(0.2)
        held = {"first": _snapshot(first), "second": _snapshot(second)}
        native_held = native_snapshot()
        assert native_held == native_at_receipt, (native_at_receipt, native_held)
        assert held == at_receipt, (before, pause, at_receipt, held)
        repeated = control(port, first.id, "pause")
        assert {k: pause["receipt"][k] for k in STATE_KEYS} == {
            k: repeated["receipt"][k] for k in STATE_KEYS}, (pause, repeated)

        # A new request must remain pending while every active slot is held.
        queued = Stream(port, tokens=16, id_slot=None)
        time.sleep(0.15)
        with queued.cv:
            assert not queued.done and queued.error is None and queued.id is None, (
                queued.done, queued.error, queued.id)

        resumed = control(port, first.id, "resume")
        paused = False
        assert {k: pause["receipt"][k] for k in STATE_KEYS} == {
            k: resumed["receipt"][k] for k in STATE_KEYS}, (pause, resumed)
        first.wait(lambda: _snapshot(first)[1] > at_receipt["first"][1])
        second.wait(lambda: _snapshot(second)[1] > at_receipt["second"][1])

        first.wait(lambda: len(first.tokens) >= 32)
        second.wait(lambda: len(second.tokens) >= 32)

        # Free the second slot so the deferred request can be admitted after resume.
        second.close()
        queued.wait(lambda: queued.id is not None, timeout=10)
        queued_result = queued.result()

        # The old stream was intentionally canceled to free slot 1. Start a fresh
        # stream before checking that a disconnected control does not strand the gate.
        second = Stream(port, id_slot=1)
        second.wait(lambda: len(second.tokens) >= 12, timeout=10)
        disconnect_before = {"first": _snapshot(first), "second": _snapshot(second)}
        _disconnect_control(port, first.id)
        first.wait(lambda: _snapshot(first)[1] > disconnect_before["first"][1], timeout=10)
        second.wait(lambda: _snapshot(second)[1] > disconnect_before["second"][1], timeout=10)
        return {
            "pause": pause,
            "held": held,
            "repeated_pause": repeated,
            "resume": resumed,
            "queued_after_resume": queued_result,
            "frozen_snapshots": {"client_at_receipt": client_at_receipt,
                "at_receipt": at_receipt, "held": held,
                "native_at_receipt": native_at_receipt, "native_held": native_held},
            "control_disconnect": {
                "before": disconnect_before,
                "after": {"first": _snapshot(first), "second": _snapshot(second)},
            },
        }
    finally:
        if paused and first.id:
            try:
                control(port, first.id, "resume")
            except BaseException:
                pass
        if queued is not None and not queued.done:
            queued.close()
        if not first.done:
            first.close()
        if not second.done:
            second.close()


def cancellation(port):
    seed(port)
    stream = Stream(port, tokens=512)
    try:
        stream.wait(lambda: len(stream.tokens) >= 12)
        p = control(port, stream.id, "pause")
        start = time.perf_counter_ns()
        stream.close()
        for _ in range(100):
            slots, _ = request(port, "/slots")
            if not any(s["is_processing"] for s in slots):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("cancelled pause owner stayed active")
        elapsed = (time.perf_counter_ns()-start)/1e6
        successor = sample(port, [])
        return {"pause": p, "disconnect_to_slot_idle_ms": elapsed,
                "successor": successor}
    finally:
        if not stream.done:
            stream.close()

def run(port, mode, output, repeats):
    output = pathlib.Path(output)
    assert not output.exists(), output
    report = {"mode": mode, "port": port, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "clock": "Linux perf_counter_ns; native receipts use ggml monotonic microseconds",
              "warmup": None, "samples": [], "status": "running"}
    def save():
        output.write_text(json.dumps(report,indent=2)+"\n")
    save()
    try:
        report["warmup"] = sample(port, [12] if mode == "pause" else [])
        save()
        expected = report["warmup"]["tokens"]
        expected_text = report["warmup"]["text_sha256"]
        if mode == "pause":
            reference = sample(port, [])
            assert reference["tokens"] == expected and reference["text_sha256"] == expected_text
            report["unpaused_reference"] = reference
            report["two_slot"] = two_slot_probe(port)
        for i in range(repeats):
            points = [12, 64, 128] if mode == "pause" else []
            value = sample(port, points)
            report["samples"].append(value)
            save()
            assert value["tokens"] == expected, ("token sequence mismatch", i)
            assert value["text_sha256"] == expected_text, ("text mismatch", i)
        if mode == "pause":
            report["cancellation"] = cancellation(port)
            save()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
    print(json.dumps({"output":str(output),"status":report["status"],"samples":len(report["samples"]),
                      "pause_samples":sum(len(s["pauses"]) for s in report["samples"])}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--mode",choices=["baseline","pause"],required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--repeats",type=int,default=6)
    args = parser.parse_args()
    run(args.port,args.mode,args.output,args.repeats)
