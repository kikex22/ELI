import time
import unittest

from eli.gpu import GPUMonitor, Sample, decode_throttle, idle_power, parse_line, parse_processes, window_stats

from fakes import FAKE_GPU_COMMAND

LINE = "37, 19, 126, 16380, 20.89, 165.00, 37, 840, 810, P5, 0x0000000000000001"


def sample(ts, power=100.0, mem=9000.0, util=80.0, temp=60.0, throttle=()):
    return Sample(ts, util, 10.0, mem, 16380.0, power, 165.0, temp, 1800.0, 8000.0, "P2", tuple(throttle))


class ParseTests(unittest.TestCase):
    def test_parse_line_reads_all_fields(self):
        s = parse_line(LINE, 10.0)
        self.assertEqual((s.util, s.mem_used, s.mem_total, s.power, s.power_limit), (37.0, 126.0, 16380.0, 20.89, 165.0))
        self.assertEqual((s.temp, s.sm_clock, s.mem_clock, s.pstate, s.throttle), (37.0, 840.0, 810.0, "P5", ()))

    def test_unsupported_fields_become_none(self):
        s = parse_line("37, 19, 126, 16380, [N/A], 165.00, 37, 840, 810, P5, 0x0", 1.0)
        self.assertIsNone(s.power)

    def test_wrong_field_count_is_ignored(self):
        self.assertIsNone(parse_line("1, 2, 3", 1.0))
        self.assertIsNone(parse_line("", 1.0))

    def test_throttle_bits(self):
        self.assertEqual(decode_throttle("0x0000000000000001"), ())  # inactiva no es frenado
        self.assertEqual(decode_throttle("0x4"), ("limite de potencia",))
        self.assertEqual(decode_throttle("0x44"), ("limite de potencia", "limite termico (hardware)"))
        self.assertEqual(decode_throttle("basura"), ())

    def test_processes(self):
        out = "1234, /usr/local/bin/ollama, 9100\n55, python3, [N/A]\nbasura\n"
        self.assertEqual(parse_processes(out), [
            {"pid": 1234, "name": "ollama", "mem_mib": 9100.0},
            {"pid": 55, "name": "python3", "mem_mib": None},
        ])


class StatsTests(unittest.TestCase):
    def test_window_stats_and_energy(self):
        samples = [sample(0.0, 100.0, 9000), sample(1.0, 120.0, 9500, util=90), sample(2.0, 140.0, 9400, temp=65)]
        stats = window_stats(samples, 0.0, 2.0, baseline_power=20.0)
        self.assertEqual(stats["vram_peak_mib"], 9500)
        self.assertEqual(stats["power_mean_w"], 120.0)
        self.assertEqual(stats["power_peak_w"], 140.0)
        self.assertEqual(stats["temp_max_c"], 65)
        self.assertAlmostEqual(stats["energy_wh"], 120.0 * 2 / 3600, places=5)
        self.assertAlmostEqual(stats["energy_net_wh"], (120.0 - 20.0) * 2 / 3600, places=5)

    def test_net_energy_never_negative(self):
        stats = window_stats([sample(0.0, 10.0), sample(1.0, 10.0)], 0.0, 1.0, baseline_power=50.0)
        self.assertEqual(stats["energy_net_wh"], 0.0)

    def test_empty_or_inverted_window(self):
        self.assertIsNone(window_stats([], 0.0, 1.0))
        self.assertIsNone(window_stats([sample(0.0)], 1.0, 1.0))

    def test_throttle_reasons_collected(self):
        stats = window_stats([sample(0.0, throttle=["a"]), sample(1.0, throttle=["b", "a"])], 0.0, 1.0)
        self.assertEqual(stats["throttle"], ["a", "b"])

    def test_idle_power_ignores_busy_samples(self):
        idle = [sample(i * 0.25, 30.0, util=5.0) for i in range(10)]
        busy = [sample(3 + i * 0.25, 120.0, util=90.0) for i in range(20)]
        self.assertEqual(idle_power(idle + busy), 30.0)

    def test_idle_power_needs_enough_idle_samples(self):
        busy = [sample(i * 0.25, 120.0, util=90.0) for i in range(40)]
        self.assertIsNone(idle_power(busy))
        self.assertIsNone(idle_power([sample(i * 0.25, 30.0, util=5.0) for i in range(5)]))


class MonitorTests(unittest.TestCase):
    def test_monitor_collects_publishes_and_windows(self):
        mon = GPUMonitor(command=FAKE_GPU_COMMAND)
        q = mon.subscribe()
        mon.start()
        try:
            first = q.get(timeout=5)
            self.assertEqual(first.mem_used, 9000.0)
            self.assertEqual(first.throttle, ("limite de potencia",))
            deadline = time.time() + 3
            while len(mon.history(60)) < 5 and time.time() < deadline:
                time.sleep(0.05)
            self.assertGreaterEqual(len(mon.history(60)), 5)
            self.assertTrue(mon.available)
            now = time.time()
            self.assertTrue(all(now - 5 <= s.ts <= now for s in mon.window(now - 5, now)))
        finally:
            mon.unsubscribe(q)
            mon.stop()

    def test_missing_nvidia_smi_reports_error(self):
        mon = GPUMonitor(command=["/no/existe/nvidia-smi"])
        mon.start()
        try:
            deadline = time.time() + 3
            while mon.error is None and time.time() < deadline:
                time.sleep(0.05)
            self.assertFalse(mon.available)
            self.assertIn("nvidia-smi", mon.error)
        finally:
            mon.stop()


if __name__ == "__main__":
    unittest.main()
