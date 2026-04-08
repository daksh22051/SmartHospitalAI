"""
API route tests for the OpenEnv FastAPI wrapper.
"""

import unittest
from fastapi.testclient import TestClient

from smart_hospital_orchestration.app import app


class TestAppAPI(unittest.TestCase):
    def test_reset_endpoint_returns_json_serializable_state(self):
        client = TestClient(app)
        response = client.post("/reset", json={"seed": 42, "task": "medium"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        self.assertIsInstance(payload.get("state"), dict)
        self.assertIsInstance(payload.get("info"), dict)
        self.assertIn("patients", payload["state"])
        self.assertIn("readable", payload["state"])

    def test_step_endpoint_returns_serializable_transition(self):
        client = TestClient(app)
        client.post("/reset", json={"seed": 42, "task": "medium"})
        response = client.post("/step", json={"action": 0})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        self.assertIsInstance(payload.get("state"), dict)
        self.assertIsInstance(payload.get("reward"), float)
        self.assertIsInstance(payload.get("done"), bool)
        self.assertIsInstance(payload.get("info"), dict)


if __name__ == "__main__":
    unittest.main()
