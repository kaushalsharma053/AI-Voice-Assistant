import unittest
from saarthi.core.intent_classifier import IntentClassifier
from saarthi.core.executor import ActionExecutor

class TestIntentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier()
        cls.executor = ActionExecutor(cls.classifier)

    def test_model_loaded(self):
        self.assertTrue(self.classifier.is_loaded, "Deep Learning model should be loaded successfully.")

    def test_greeting_intent(self):
        result = self.classifier.predict("Hello Saarthi")
        self.assertEqual(result["detected_intent"], "greeting")
        self.assertGreaterEqual(result["confidence"], 50.0)

    def test_open_app_intent(self):
        result = self.classifier.predict("open notepad")
        self.assertEqual(result["detected_intent"], "open_app")

    def test_system_status_intent(self):
        result = self.classifier.predict("check cpu usage")
        self.assertEqual(result["detected_intent"], "system_status")

    def test_time_intent(self):
        result = self.classifier.predict("what time is it")
        self.assertEqual(result["detected_intent"], "get_time")

    def test_executor_flow(self):
        result = self.executor.process_query("what is today's date")
        self.assertEqual(result["intent"], "get_date")
        self.assertIn("Today is", result["response_text"])

    def test_safe_math_calculation(self):
        result = self.executor.process_query("calculate 15 plus 25")
        self.assertIn("40", result["response_text"])

if __name__ == '__main__':
    unittest.main()
