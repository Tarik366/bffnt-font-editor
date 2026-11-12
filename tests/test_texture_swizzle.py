import unittest
import struct

from PIL import Image

from bffnt_editor.texture.decoder import (
    SwizzleMode,
    decode_bntx_sheet,
    deswizzle_block_linear,
    get_format_info,
    parse_bntx,
    swap_switch2_gob_sectors,
)
from bffnt_editor.texture.encoder import (
    BNTXBuildParams,
    build_bntx,
    swizzle_block_linear,
)


class TextureSwizzleTests(unittest.TestCase):
    def test_existing_switch_block_height_round_trip(self):
        width = height = 64
        linear_data = bytes(range(256)) * 16

        swizzled = swizzle_block_linear(width, height, 1, 1, 1, linear_data)
        deswizzled = deswizzle_block_linear(width, height, 1, 1, 1, swizzled)

        self.assertEqual(deswizzled, linear_data)

    def test_explicit_block_height_round_trip(self):
        width = height = 64
        linear_data = bytes(range(256)) * 16
        block_height = 2

        swizzled = swizzle_block_linear(
            width, height, 1, 1, 1, linear_data,
            block_height_override=block_height,
        )
        deswizzled = deswizzle_block_linear(
            width, height, 1, 1, 1, swizzled,
            block_height_override=block_height,
        )

        self.assertEqual(deswizzled, linear_data)

    def test_bc4_explicit_block_height_round_trip(self):
        width = height = 512
        bpp, block_width, block_height = get_format_info(0x1D01)
        width_in_blocks = width // block_width
        height_in_blocks = height // block_height
        linear_data = bytes(range(256)) * (
            width_in_blocks * height_in_blocks * bpp // 256
        )

        swizzled = swizzle_block_linear(
            width, height, block_width, block_height, bpp, linear_data,
            block_height_override=2,
        )
        deswizzled = deswizzle_block_linear(
            width, height, block_width, block_height, bpp, swizzled,
            block_height_override=2,
        )

        self.assertEqual(deswizzled, linear_data)

    def test_byte_coordinate_round_trip_with_gob_crossing_pixels(self):
        width, height = 37, 17
        bpp, block_width, block_height = get_format_info(0x0301)
        linear_data = bytes(
            (index * 37) % 256
            for index in range(width * height * bpp)
        )

        swizzled = swizzle_block_linear(
            width, height, block_width, block_height, bpp, linear_data,
            block_height_override=4,
            byte_coordinates=True,
        )
        deswizzled = deswizzle_block_linear(
            width, height, block_width, block_height, bpp, swizzled,
            block_height_override=4,
            byte_coordinates=True,
        )

        self.assertEqual(deswizzled, linear_data)

    def test_switch2_desktop_gob_sector_swap(self):
        data = bytes(range(256))
        expected = (
            data[:32] + data[64:96] + data[32:64] + data[96:128]
            + data[128:160] + data[192:224] + data[160:192] + data[224:]
        )

        swapped = swap_switch2_gob_sectors(data)

        self.assertEqual(swapped, expected)
        self.assertEqual(swap_switch2_gob_sectors(swapped), data)

    def test_switch2_bntx_reads_block_height_from_correct_field(self):
        width = height = 64
        rgba_data = bytes(
            channel
            for y in range(height)
            for x in range(width)
            for channel in (255, 255, 255, (x * 7 + y * 13) % 256)
        )
        image = Image.frombytes("RGBA", (width, height), rgba_data)
        bntx_data = build_bntx(
            [image],
            BNTXBuildParams(
                format_code=0x0201,
                swizzle_mode=SwizzleMode.SWITCH2,
            ),
        )

        bntx_data = bytearray(bntx_data)
        brti_offset = bntx_data.find(b"BRTI")
        struct.pack_into("<I", bntx_data, brti_offset + 0x38, 1)
        texture = parse_bntx(bytes(bntx_data))[0]
        self.assertEqual(texture.size_range, 3)
        decoded = decode_bntx_sheet(texture, 0, SwizzleMode.SWITCH2)

        self.assertEqual(decoded.getchannel("A").tobytes(), image.getchannel("A").tobytes())

    def test_switch2_desktop_layout_round_trip_with_gob_crossing_pixels(self):
        width, height = 37, 17
        bpp, block_width, block_height = get_format_info(0x0301)
        linear_data = bytes(
            (index * 37) % 256
            for index in range(width * height * bpp)
        )

        switch_data = swizzle_block_linear(
            width, height, block_width, block_height, bpp, linear_data,
            block_height_override=4,
        )
        switch2_data = swap_switch2_gob_sectors(swizzle_block_linear(
            width, height, block_width, block_height, bpp, linear_data,
            block_height_override=4,
            byte_coordinates=True,
        ))
        self.assertNotEqual(switch_data, switch2_data)
        self.assertEqual(
            deswizzle_block_linear(
                width, height, block_width, block_height, bpp,
                swap_switch2_gob_sectors(switch2_data),
                block_height_override=4,
                byte_coordinates=True,
            ),
            linear_data,
        )


if __name__ == "__main__":
    unittest.main()
