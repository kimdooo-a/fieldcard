"""Offline tests: no model, no network. Run with `python -m unittest discover tests`."""
import datetime as dt
import socket
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fieldcard as fc

GOOD = {
    "title": "Riverside Autumn Watch",
    "items": [
        {"find": "A cluster of red maple leaves on a low branch", "why": "Maples pull green chlorophyll back in autumn.", "proof": "sketch"},
        {"find": "Wet dark stones at the water's edge", "why": "Algae films make river stones darker and slick.", "proof": "photo"},
        {"find": "How many different bird calls in five minutes", "why": "Autumn flocks mix species and call to stay together.", "proof": "tally"},
        {"find": "Seed heads on tall riverside grasses", "why": "Grasses ripen seed in late summer and autumn.", "proof": "photo"},
        {"find": "Ridged grey bark of a willow", "why": "Willow bark furrows deepen as the trunk widens.", "proof": "sketch"},
    ],
    "sit_spot": "Sit on the steps by the water for one quiet minute.",
}


class SkyMath(unittest.TestCase):
    def test_seasons_flip_by_hemisphere(self):
        d = dt.date(2026, 10, 6)
        self.assertEqual(fc.season_for(d, 37.5), "autumn")
        self.assertEqual(fc.season_for(d, -33.9), "spring")

    def test_seoul_sunset_is_plausible(self):
        # Seoul, 2026-10-06: api.sunrise-sunset.org gives 18:10:20 KST.
        _, sunset = fc.sun_times(dt.date(2026, 10, 6), 37.528, 126.934, 9)
        minutes = sunset.hour * 60 + sunset.minute
        self.assertTrue(18 * 60 + 8 <= minutes <= 18 * 60 + 12, sunset)

    def test_polar_night_returns_none(self):
        self.assertEqual(fc.sun_times(dt.date(2026, 12, 21), 80.0, 15.0, 1), (None, None))


class CardRules(unittest.TestCase):
    def test_good_card_passes(self):
        self.assertEqual(fc.validate_quest(GOOD), [])

    def test_unsafe_items_are_rejected(self):
        for bad in ["Pick a mushroom and taste it", "Feed the ducks some bread",
                    "Climb a tree to see a nest", "Check the bird app on your phone"]:
            card = {**GOOD, "items": [{**GOOD["items"][0], "find": bad}] + GOOD["items"][1:]}
            self.assertTrue(fc.validate_quest(card), bad)

    def test_harmless_words_are_not_rejected(self):
        card = {**GOOD, "items": [{**GOOD["items"][0], "find": "Ivy climbing a brick wall"}] + GOOD["items"][1:]}
        self.assertEqual(fc.validate_quest(card), [])

    def test_tally_must_say_what_to_count(self):
        sound = {**GOOD["items"][2], "find": "The crunch of dry leaves underfoot"}
        self.assertTrue(fc.validate_quest({**GOOD, "items": GOOD["items"][:2] + [sound] + GOOD["items"][3:]}))

    def test_wrong_count_and_duplicates(self):
        self.assertTrue(fc.validate_quest({**GOOD, "items": GOOD["items"][:4]}))
        self.assertTrue(fc.validate_quest({**GOOD, "items": GOOD["items"][:4] + [GOOD["items"][0]]}))


class Parsing(unittest.TestCase):
    def test_parses_fenced_json(self):
        self.assertEqual(fc.parse_json_reply('```json\n{"a": 1}\n```'), {"a": 1})

    def test_parses_json_inside_prose(self):
        self.assertEqual(fc.parse_json_reply('Sure! {"a": 2} hope that helps'), {"a": 2})


class Airplane(unittest.TestCase):
    def test_blocks_the_internet_but_not_localhost(self):
        fc.enable_airplane_mode()
        try:
            with self.assertRaises(Exception) as ctx:
                urllib.request.urlopen("https://example.com", timeout=5)
            self.assertIn("airplane", str(ctx.exception))
            # loopback lookups are still allowed
            self.assertTrue(socket.getaddrinfo("127.0.0.1", 11434))
        finally:
            socket.getaddrinfo = fc._real_getaddrinfo
            socket.create_connection = fc._real_create_connection


class Rendering(unittest.TestCase):
    def test_card_html_escapes_model_text(self):
        card = {
            "fieldcard": fc.__version__, "title": "<script>x</script>", "sit_spot": "sit",
            "items": [{"n": i + 1, **it} for i, it in enumerate(GOOD["items"])],
            "context": {"date": "2026-10-06", "season": "autumn", "place": "park", "minutes": 30, "sunset": "18:09"},
            "generated_by": {"model": "gemma4:e4b"},
        }
        page = fc.render_card_html(card)
        self.assertNotIn("<script>", page)
        self.assertIn("sunset 18:09", page)


if __name__ == "__main__":
    unittest.main()
