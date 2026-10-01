"""Tests for shuffle byte accounting."""

import os
import shutil
import tempfile
import unittest

from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler
from backend.master.shuffle import ShuffleCoordinator
from backend.worker.executor import _run_map
from backend.worker.shuffle_store import ShuffleStore


class TestShuffleBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.shuffle = ShuffleCoordinator(
            self.storage, self.jm, self.registry, self.logbus
        )
        self.worker = self.registry.register({
            "worker_id": "w-test",
            "name": "test-worker",
            "host": "127.0.0.1",
            "port": 9000,
            "cpu_cores": 1,
            "mem_total_mb": 128,
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_map_reports_actual_partition_bytes(self):
        records = ["map shuffle bytes", "map shuffle bytes"]
        spec = {
            "task_id": "m-0000",
            "job_id": "job",
            "kind": "map",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "params": {},
            "partition_count": 2,
            "records": records,
            "spill_records": 10,
        }

        result = _run_map(spec, self.tmp, lambda progress, processed, emitted: None)
        store = ShuffleStore(self.tmp)

        self.assertEqual(
            result["partition_sizes"],
            store.partition_sizes("job", "m-0000"),
        )
        self.assertTrue(all(size > 0 for size in result["partition_sizes"].values()))

    def test_coordinator_uses_map_partition_sizes(self):
        job = self.jm.submit({
            "name": "t",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 1,
            "num_reduce_tasks": 1,
            "input_rows": 1,
            "params": {},
        })
        map_task = self.jm.tasks_for(job.job_id, "map")[0]
        map_task.worker_id = self.worker.worker_id

        path = os.path.join(self.tmp, "shuffle", job.job_id, map_task.task_id, "part-0000.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * 128)
        map_task.stats["partition_sizes"] = {"part-0000": 128}
        self.jm.save_task(job.job_id, map_task)

        self.shuffle.build(job)
        matrix = self.shuffle.matrix(job)

        self.assertEqual(matrix["total_bytes"], 128)
        self.assertEqual(matrix["partitions"][0]["total_bytes"], 128)
        self.assertEqual(matrix["partitions"][0]["sources"][0]["bytes"], 128)

    def test_scheduler_persists_partition_sizes(self):
        job = self.jm.submit({
            "name": "t",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 1,
            "num_reduce_tasks": 1,
            "input_rows": 1,
            "params": {},
        })
        task = self.jm.tasks_for(job.job_id, "map")[0]
        task.worker_id = self.worker.worker_id
        self.jm.save_task(job.job_id, task)

        scheduler = Scheduler(
            self.storage,
            self.jm,
            self.registry,
            self.shuffle,
            FaultTolerance(self.storage, self.jm, self.config, self.logbus),
            Metrics(self.storage),
            self.config,
            self.logbus,
        )
        scheduler.on_task_complete({
            "worker_id": self.worker.worker_id,
            "job_id": job.job_id,
            "task_id": task.task_id,
            "kind": "map",
            "status": "SUCCEEDED",
            "records_processed": 1,
            "records_emitted": 1,
            "duration_ms": 1,
            "partition_sizes": {"part-0000": 128},
            "results": [],
        })

        saved = self.jm.get_task(job.job_id, task.task_id)
        self.assertEqual(saved.stats["partition_sizes"], {"part-0000": 128})
        self.assertNotIn("partition_size_entries", saved.stats)


if __name__ == "__main__":
    unittest.main()
