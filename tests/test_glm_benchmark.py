import unittest

import glm_benchmark


class GlmBenchmarkTests(unittest.TestCase):
    def test_health_url_uses_server_root(self):
        self.assertEqual(
            glm_benchmark._health_url("http://localhost:11111/v1"),
            "http://localhost:11111/health",
        )
        self.assertEqual(
            glm_benchmark._health_url("http://localhost:11112"),
            "http://localhost:11112/health",
        )

    def test_discover_model_uses_first_model_id(self):
        original = glm_benchmark._request_json
        glm_benchmark._request_json = lambda method, url, **kwargs: {
            "data": [{"id": "glm-test"}, {"id": "backup"}]
        }
        try:
            self.assertEqual(
                glm_benchmark.discover_model("http://localhost:11111/v1", timeout=30),
                "glm-test",
            )
        finally:
            glm_benchmark._request_json = original

    def test_extract_output_text_handles_text_parts(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "ignored"}},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(glm_benchmark._extract_output_text(payload), "hello world")

    def test_summarize_runs_calculates_average_rates(self):
        runs = [
            glm_benchmark.RunMetrics(
                latency_seconds=2.0,
                prompt_tokens=120,
                completion_tokens=60,
                total_tokens=180,
                output_chars=240,
            ),
            glm_benchmark.RunMetrics(
                latency_seconds=3.0,
                prompt_tokens=150,
                completion_tokens=90,
                total_tokens=240,
                output_chars=360,
            ),
        ]

        summary = glm_benchmark.summarize_runs(
            target="litellm",
            base_url="http://localhost:11111/v1",
            model="glm-test",
            warmup_runs=1,
            measured_runs=runs,
        )

        self.assertEqual(summary.avg_latency_seconds, 2.5)
        self.assertEqual(summary.min_latency_seconds, 2.0)
        self.assertEqual(summary.max_latency_seconds, 3.0)
        self.assertEqual(summary.avg_prompt_tokens, 135.0)
        self.assertEqual(summary.avg_completion_tokens, 75.0)
        self.assertEqual(summary.avg_total_tokens, 210.0)
        self.assertEqual(summary.avg_completion_tokens_per_second, 30.0)
        self.assertEqual(summary.avg_total_tokens_per_second, 85.0)


if __name__ == "__main__":
    unittest.main()
