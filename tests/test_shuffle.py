"""Regression test: the shuffle matrix must report real map-output bytes.

Covers the full accounting chain — worker map stats -> scheduler task stats
-> ShuffleCoordinator matrix — which previously zeroed out every byte count
(unit conversion, mismatched dict keys and a spurious per-source increment).
"""

import shutil
import tempfile
import unittest

from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.registry import WorkerRegistry
from backend.master.shuffle import ShuffleCoordinator
from backend.tasks.samples import generate_input_records
from backend.worker.executor import _run_map
from backend.worker.shuffle_store import ShuffleStore


class TestShuffleByteAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.logbus = LogBus(self.storage)
        self.config = ClusterConfig()
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.worker = self.registry.register({
            "worker_id": "w-1", "name": "w-1", "host": "127.0.0.1",
            "port": 9001, "cpu_cores": 4, "mem_total_mb": 1024,
        })
        self.coord = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_job_maps(self, num_maps=3, num_reduces=2, rows=500):
        job = self.jm.submit({
            "name": "wc", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": num_maps, "num_reduce_tasks": num_reduces,
            "input_rows": rows, "params": {},
        })
        records = generate_input_records("wordcount", rows, seed=1)
        map_tasks = self.jm.tasks_for(job.job_id, "map")
        chunk = (len(records) + len(map_tasks) - 1) // len(map_tasks)
        for i, mt in enumerate(map_tasks):
            spec = {
                "mapper": "wordcount_mapper",
                "records": records[i * chunk:(i + 1) * chunk],
                "partition_count": num_reduces,
                "job_id": job.job_id,
                "task_id": mt.task_id,
            }
            result = _run_map(spec, self.tmp, lambda *a: None)
            sizes = result["partition_sizes"]

            def apply(t, sizes=sizes):
                t.worker_id = self.worker.worker_id
                stats = dict(t.stats or {})
                stats["partition_sizes"] = sizes
                t.stats = stats

            self.jm.apply_task(job.job_id, mt.task_id, apply)
        return job, map_tasks

    def test_matrix_bytes_match_on_disk_output(self):
        job, map_tasks = self._run_job_maps()
        self.coord.build(job)
        matrix = self.coord.matrix(job)

        store = ShuffleStore(self.tmp)
        expected = sum(
            size
            for mt in map_tasks
            for size in store.partition_sizes(job.job_id, mt.task_id).values()
        )
        self.assertGreater(expected, 0)
        self.assertEqual(matrix["total_bytes"], expected)
        for part in matrix["partitions"]:
            self.assertEqual(
                part["total_bytes"],
                sum(s["bytes"] for s in part["sources"]),
            )

    def test_map_stats_are_bytes_not_kib(self):
        job, map_tasks = self._run_job_maps(num_maps=1, num_reduces=1, rows=50)
        mt = self.jm.tasks_for(job.job_id, "map")[0]
        reported = (mt.stats or {}).get("partition_sizes", {})
        on_disk = ShuffleStore(self.tmp).partition_sizes(job.job_id, mt.task_id)
        self.assertEqual(reported, on_disk)


if __name__ == "__main__":
    unittest.main()
