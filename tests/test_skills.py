import unittest
from saarthi.skills import system_skills, web_skills, info_skills, productivity_skills, chat_skills

class TestSkillsSubsystem(unittest.TestCase):
    def test_system_status(self):
        status = system_skills.get_system_status()
        self.assertIsInstance(status, str)
        self.assertIn("CPU", status)
        self.assertIn("RAM", status)

    def test_current_time(self):
        t = info_skills.get_current_time()
        self.assertIn("The current time is", t)

    def test_current_date(self):
        d = info_skills.get_current_date()
        self.assertIn("Today is", d)

    def test_math_calculation(self):
        # Basic arithmetic
        res = info_skills.calculate_expression("what is 100 divided by 4")
        self.assertIn("25", res)

        # Complex expression
        res2 = info_skills.calculate_expression("calculate (12 * 5) + 40")
        self.assertIn("100", res2)

        # Square root
        res3 = info_skills.calculate_expression("what is square root of 144")
        self.assertIn("12", res3)

    def test_notes_skill(self):
        # Write note
        write_res = productivity_skills.take_note("test note for unit testing")
        self.assertIn("Note recorded", write_res)

        # Read note
        read_res = productivity_skills.read_notes()
        self.assertIn("test note", read_res)

        # Clear note
        clear_res = productivity_skills.clear_notes()
        self.assertIn("cleared", clear_res)

    def test_chat_and_viva(self):
        viva_ans = chat_skills.get_viva_explanation("explain backpropagation")
        self.assertIn("gradient", viva_ans.lower())

        quote = chat_skills.get_motivational_quote()
        self.assertTrue(len(quote) > 10)

if __name__ == '__main__':
    unittest.main()
