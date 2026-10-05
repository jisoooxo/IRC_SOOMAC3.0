import unittest
from io import BytesIO
from PIL import Image
from soomac_irc.vlm import build_vlm_request
from soomac_irc.vlm_ui import build_vlm_ui_jpeg


class TestVlmUi(unittest.TestCase):
    def test_three_panels_and_model_inputs_are_preserved(self):
        ref, prev, current = [Image.new("RGB", (80, 60), color)
                              for color in ("red", "green", "blue")]
        request = build_vlm_request("양파", [current] * 5, ref, prev, "pass")
        self.assertEqual(len(request["images"]), 7)
        self.assertIs(request["images"][0], ref)
        self.assertIs(request["images"][1], prev)
        image = Image.open(BytesIO(build_vlm_ui_jpeg(request)))
        self.assertEqual(image.size, (1440, 396))
        for index, channel in enumerate((0, 1, 2)):
            pixel = image.getpixel((index * 480 + 240, 216))
            self.assertEqual(pixel.index(max(pixel)), channel)

    def test_first_task_and_lid_have_explicit_empty_panels(self):
        frame = Image.new("RGB", (40, 80))
        first = build_vlm_request("얇은면", [frame], frame)
        self.assertIsNone(first["ui_panels"][1]["image"])
        lid = build_vlm_request("뚜껑", [frame], comparison_image=frame)
        self.assertIsNone(lid["ui_panels"][0]["image"])
        self.assertTrue(build_vlm_ui_jpeg(first))
        self.assertTrue(build_vlm_ui_jpeg(lid))

    def test_retry_is_not_labelled_success(self):
        frame = Image.new("RGB", (40, 40))
        request = build_vlm_request("양파", [frame], frame, frame, "retried")
        self.assertIn("NOT VERIFIED", request["ui_panels"][1]["label"])
