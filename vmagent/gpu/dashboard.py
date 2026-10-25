"""Local upload, run and monitoring UI: python -m vmagent.gpu.dashboard."""
import argparse
import json
import mimetypes
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .environment import DEFAULT_NODES, DEFAULT_TASKS, ROOT, load_nodes, load_workload
from .run import make_environment

STATIC = Path(__file__).with_name("static")
RESULT_FILES = {"summary.json", "tasks.csv", "events.csv", "resource_history.csv", "nodes_final.csv", "run_manifest.json"}


class DashboardState:
    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.uploads, self.runs = {}, {}
        self.current_id = None

    def upload(self, kind, filename, content):
        if kind not in ("nodes", "tasks") or not isinstance(filename, str) or not isinstance(content, str):
            raise ValueError("Upload kind, filename and text content are required")
        suffix = Path(filename).suffix.lower()
        if suffix not in (".csv", ".json"):
            raise ValueError("Upload a UTF-8 .csv or .json file")
        token = uuid.uuid4().hex
        path = self.output_root/"uploads"/(token+suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        try:
            count = len(load_nodes(path)) if kind == "nodes" else len(load_workload(path)[0])
        except Exception:
            path.unlink()
            raise
        with self.lock:
            self.uploads[token] = {"path": path, "kind": kind, "filename": filename, "count": count}
        return {"upload_id": token, "filename": filename, "record_count": count, "kind": kind}

    def launch(self, config):
        with self.lock:
            if self.current_id and self.runs[self.current_id]["status"] in ("starting", "running", "stopping"):
                raise ValueError("An experiment is already running; stop it or wait for completion")
            paths = {}
            for kind, default in [("nodes", DEFAULT_NODES), ("tasks", DEFAULT_TASKS)]:
                token = config.get(kind+"_upload_id")
                upload = self.uploads.get(token) if token else None
                if token and (upload is None or upload["kind"] != kind):
                    raise ValueError("Invalid upload ID for " + kind)
                paths[kind] = upload["path"] if upload else default
            policy = config.get("policy", "first_fit")
            if policy not in ("first_fit", "best_fit", "worst_fit", "random"):
                raise ValueError("Unknown scheduling policy")
            options = {key: config.get(key, default) for key, default in
                       [("clock", "realtime"), ("speed", 1.), ("queue_policy", "backfill"),
                        ("max_sim_time", 600.), ("cpu_background_fraction", 0.), ("seed", 42)]}
            if "measurement_seconds" in config:
                start = float(config.get("measurement_start_seconds", 0.))
                options["measurement_window"] = [start, start+float(config["measurement_seconds"])]
            env, metadata = make_environment(paths["nodes"], paths["tasks"], **options)
            run_id = uuid.uuid4().hex
            folder = self.output_root/"runs"/run_id
            manifest = {"nodes": str(paths["nodes"].resolve()), "tasks": str(paths["tasks"].resolve()),
                        "policy": policy, "options": options, "workload_metadata": metadata}
            run = {"status": "starting", "env": env, "output": folder, "error": None, "manifest": manifest}
            self.runs[run_id], self.current_id = run, run_id

            def worker():
                try:
                    run["status"] = "running"
                    env.run(policy)
                    env.export(folder)
                    (folder/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    run["status"] = "cancelled" if env.end_reason == "cancelled" else "completed"
                except Exception as exc:
                    run["error"] = str(exc)
                    run["status"] = "failed"
                    traceback.print_exc()
            threading.Thread(target=worker, daemon=True).start()
            return {"run_id": run_id}

    def snapshot(self):
        with self.lock:
            if self.current_id is None:
                return {"status": "idle"}
            run = self.runs[self.current_id]
            result = {"status": run["status"], "run_id": self.current_id, "error": run["error"],
                      "output_folder": str(run["output"].resolve())}
            if run["env"].initialized:
                result.update(run["env"].snapshot())
            return result


def handler_class(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Polling requests otherwise fill the terminal every half second.
            if args and str(args[0]).startswith("POST"):
                super().log_message(format, *args)

        def send_bytes(self, data, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", 'attachment; filename="{}"'.format(filename))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/config":
                self.send_json({"default_nodes": str(DEFAULT_NODES), "default_tasks": str(DEFAULT_TASKS),
                                "defaults_available": DEFAULT_NODES.is_file() and DEFAULT_TASKS.is_file()})
            elif path == "/api/state":
                self.send_json(state.snapshot())
            elif path.startswith("/api/download/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[3] not in state.runs or parts[4] not in RESULT_FILES:
                    return self.send_json({"error": "Unknown output file"}, 404)
                run = state.runs[parts[3]]
                file = run["output"]/parts[4]
                if run["status"] in ("starting", "running", "stopping") or not file.is_file():
                    return self.send_json({"error": "Outputs become available after the run stops"}, 409)
                self.send_bytes(file.read_bytes(), "application/octet-stream", filename=file.name)
            elif path in ("/", "/app.js", "/style.css"):
                file = STATIC/("index.html" if path == "/" else path.lstrip("/"))
                content_type = mimetypes.guess_type(file.name)[0] or "text/plain"
                self.send_bytes(file.read_bytes(), content_type+"; charset=utf-8")
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                # No CORS; reject foreign browser origins for local mutations.
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.send_json({"error": "Foreign origin rejected"}, 403)
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 12*1024*1024:
                    return self.send_json({"error": "Request must be between 1 byte and 12 MiB"}, 413)
                config = json.loads(self.rfile.read(length))
                if not isinstance(config, dict):
                    raise ValueError("Request must contain a JSON object")
                path = urlparse(self.path).path
                if path == "/api/upload":
                    self.send_json(state.upload(config.get("kind"), config.get("filename"), config.get("content")))
                elif path == "/api/run":
                    self.send_json(state.launch(config))
                elif path == "/api/stop":
                    with state.lock:
                        if state.current_id:
                            run = state.runs[state.current_id]
                            if run["status"] in ("starting", "running"):
                                run["status"] = "stopping"
                                run["env"].stop_requested.set()
                    self.send_json({"ok": True})
                else:
                    self.send_json({"error": "Not found"}, 404)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local VMAgent GPU scheduling dashboard")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/gpu/dashboard")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler_class(DashboardState(args.output)))
    print("Open http://{}:{} (keep this terminal open; Ctrl+C to stop)".format(args.host, server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
