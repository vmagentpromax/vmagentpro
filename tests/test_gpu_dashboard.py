import csv
import io
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vmagent.gpu.dashboard import DashboardState, handler_class
from vmagent.gpu.environment import DEFAULT_NODES, DEFAULT_TASKS


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = DashboardState(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:{}".format(self.server.server_port)

    def tearDown(self):
        if self.state.current_id:
            self.state.runs[self.state.current_id]["env"].stop_requested.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, path, data=None):
        request = Request(self.url+path, data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json"} if data is not None else {})
        with urlopen(request, timeout=3) as response:
            return response.read()

    def get_json(self, path, data=None):
        return json.loads(self.request(path, data))

    def wait_finished(self):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = self.get_json("/api/state")
            if state["status"] not in ("starting", "running", "stopping"):
                return state
            time.sleep(.01)
        self.fail("Dashboard experiment did not stop")

    def test_page_and_static_resources(self):
        self.assertIn("上传服务器表", self.request("/").decode("utf-8"))
        self.assertIn(b"/api/upload", self.request("/app.js"))
        self.assertIn(b".layout", self.request("/style.css"))
        self.assertTrue(self.get_json("/api/config")["defaults_available"])

    def test_nonzero_measurement_start_excludes_warmup(self):
        self.get_json("/api/run", {"clock": "fast", "measurement_start_seconds": 24,
                                  "measurement_seconds": 96})
        final = self.wait_finished()
        self.assertEqual(final["summary"]["measurement_window_sec"], [24., 120.])
        self.assertEqual(final["summary"]["measurement_denominator_sec"], 96.)

    def test_upload_validation_run_download_and_cancel(self):
        nodes = self.get_json("/api/upload", {"kind": "nodes", "filename": DEFAULT_NODES.name,
                              "content": DEFAULT_NODES.read_text(encoding="utf-8-sig")})
        csv_path = DEFAULT_TASKS.with_suffix(".csv")
        tasks = self.get_json("/api/upload", {"kind": "tasks", "filename": csv_path.name,
                              "content": csv_path.read_text(encoding="utf-8-sig")})
        self.assertEqual(nodes["record_count"], 20)
        self.assertEqual(tasks["record_count"], 50)
        duplicate = json.loads(DEFAULT_TASKS.read_text(encoding="utf-8"))["tasks"][:1]*2
        with self.assertRaises(HTTPError) as error:
            self.request("/api/upload", {"kind": "tasks", "filename": "duplicate.json", "content": json.dumps(duplicate)})
        self.assertEqual(error.exception.code, 400)
        self.assertIn(b"Duplicate pod_id", error.exception.read())
        config = {"nodes_upload_id": nodes["upload_id"], "tasks_upload_id": tasks["upload_id"], "clock": "fast"}
        run = self.get_json("/api/run", config)
        final = self.wait_finished()
        self.assertEqual(final["summary"]["completed"], 50)
        self.assertEqual(final["summary"]["unfinished"], 0)
        self.assertTrue(all(abs(n["free_gpu"]-8) < 1e-8 and abs(n["free_cpu_cores"]-192) < 1e-8 for n in final["nodes"]))
        raw = self.request("/api/download/{}/tasks.csv".format(run["run_id"])).decode("utf-8-sig")
        output = list(csv.DictReader(io.StringIO(raw)))
        self.assertEqual(len(output), 50)
        self.assertEqual(len({row["pod_id"] for row in output}), 50)
        self.assertTrue(all(row["status"] == "completed" for row in output))
        with self.assertRaises(HTTPError):
            self.request("/api/download/{}/../summary.json".format(run["run_id"]))
        self.get_json("/api/run", {"clock": "realtime"})
        self.get_json("/api/stop", {})
        stopped = self.wait_finished()
        self.assertEqual(stopped["status"], "cancelled")
        self.assertGreater(stopped["summary"]["unfinished"], 0)
        self.assertEqual(stopped["summary"]["task_count"], 50)


if __name__ == "__main__":
    unittest.main()
