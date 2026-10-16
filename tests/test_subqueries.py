import unittest

from eli.emb import subqueries
from eli.emb.subqueries import SubqueryError, build_messages, parse_subqueries


class ParseTests(unittest.TestCase):
    def test_plain_json_array(self):
        self.assertEqual(parse_subqueries('["uno", "dos", "tres"]'), ["uno", "dos", "tres"])

    def test_think_block_and_code_fence_are_ignored(self):
        text = '<think>pienso mucho [no es json]</think>\n```json\n["a b c", "d e f"]\n```'
        self.assertEqual(parse_subqueries(text), ["a b c", "d e f"])

    def test_unclosed_think_prefix_is_dropped_when_closing_tag_exists(self):
        self.assertEqual(parse_subqueries('razonamiento sin cerrar </think> ["x y", "z w"]'), ["x y", "z w"])

    def test_bullets_and_numbers_are_accepted_when_there_is_no_json(self):
        text = "1. Donde se abre la camara\n2) Como se aplica el NMS\n- Que tracker se usa"
        self.assertEqual(parse_subqueries(text), ["Donde se abre la camara", "Como se aplica el NMS", "Que tracker se usa"])

    def test_duplicates_blank_and_too_long_items_are_dropped(self):
        items = ['["Uno", "uno", "  ", "' + "x" * 300 + '", "dos"]']
        self.assertEqual(parse_subqueries(items[0]), ["Uno", "dos"])

    def test_limit_is_applied(self):
        out = parse_subqueries('["a1 a1", "b2 b2", "c3 c3", "d4 d4"]', max_n=3)
        self.assertEqual(out, ["a1 a1", "b2 b2", "c3 c3"])

    def test_fewer_than_two_usable_subqueries_is_an_error(self):
        with self.assertRaises(SubqueryError):
            parse_subqueries('["solo una"]')
        with self.assertRaises(SubqueryError):
            parse_subqueries("")


class PromptTests(unittest.TestCase):
    def test_prompt_asks_for_search_phrases_not_questions(self):
        msgs = build_messages("Como se calcula la distancia?", 6, "en")
        system, user = msgs[0]["content"], msgs[1]["content"]
        self.assertIn("SEARCH PHRASES", system)
        self.assertIn("NO question marks", system)
        self.assertIn("6 search phrases", system)
        self.assertIn("English code vocabulary", system)
        self.assertIn("JSON array", system)
        self.assertIn("Como se calcula la distancia?", user)
        self.assertIn("English", system)
        self.assertIn("the same language as the question", build_messages("q", 3, "same")[0]["content"])

    def test_question_marks_are_removed_from_the_output(self):
        self.assertEqual(parse_subqueries('["¿Qué sensor de profundidad?", "depth sensor driver"]'),
                         ["Qué sensor de profundidad", "depth sensor driver"])

    def test_limits_are_consistent(self):
        self.assertLessEqual(subqueries.MIN_SUBQUERIES, subqueries.DEFAULT_SUBQUERIES)
        self.assertLessEqual(subqueries.DEFAULT_SUBQUERIES, subqueries.MAX_SUBQUERIES)
        self.assertEqual(subqueries.OPTIONS["temperature"], 0)
        self.assertEqual(subqueries.OPTIONS["seed"], 42)


if __name__ == "__main__":
    unittest.main()


class RefineTests(unittest.TestCase):
    def test_done_and_phrases_are_read_from_json(self):
        self.assertEqual(subqueries.parse_refine('{"done": true, "queries": []}'), (True, []))
        self.assertEqual(subqueries.parse_refine('<think>x</think>{"done": false, "queries": ["depth to distance", "camera intrinsics"]}'),
                         (False, ["depth to distance", "camera intrinsics"]))

    def test_phrases_are_capped_and_cleaned(self):
        done, phrases = subqueries.parse_refine('{"done": false, "queries": ["a b c", "d e f", "g h i", "j k l", "a b c"]}')
        self.assertEqual(phrases, ["a b c", "d e f", "g h i"])

    def test_broken_answers_raise(self):
        for bad in ("no json", "[1, 2]", '{"done": false, "queries": "uno"', ""):
            with self.subTest(bad=bad):
                with self.assertRaises(SubqueryError):
                    subqueries.parse_refine(bad)

    def test_refine_prompt_shows_files_and_asks_for_json(self):
        msgs = subqueries.build_refine_messages("como funciona", [("src/a.cpp", "void f() {}")])
        self.assertIn("src/a.cpp", msgs[1]["content"])
        self.assertIn('"done"', msgs[0]["content"])
        self.assertIn("NO question marks", msgs[0]["content"])
