import unittest
from fastapi.testclient import TestClient
from server import app

class TestFastAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_serve_index_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("SAARTHI AI", response.text)
        self.assertIn("Neural Telemetry", response.text)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["model_loaded"])

    def test_system_status_endpoint(self):
        response = self.client.get("/api/system-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cpu_percent", data)
        self.assertIn("ram_percent", data)

    def test_chat_endpoint_greeting(self):
        response = self.client.post("/api/chat", json={"query": "Hello Saarthi"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent"], "greeting")
        self.assertGreater(data["confidence"], 50.0)

    def test_chat_endpoint_math(self):
        response = self.client.post("/api/chat", json={"query": "Calculate 25 * 4"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("100", data["response_text"])

    def test_model_info_endpoint(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["architecture"], "Bidirectional LSTM (Bi-LSTM)")
        self.assertEqual(data["classes_count"], 29)

if __name__ == "__main__":
    unittest.main()
