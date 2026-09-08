# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QsaSidecarOwnerSource(unittest.TestCase):
    def test_authentication_precedes_device_allocation_and_publication(self):
        source = (ROOT / "src/attention/qsa_sidecar_owner.cc").read_text()
        self.assertLess(source.index("read_exact(path)"), source.index("cudaMalloc"))
        self.assertLess(source.index("payload hash changed"), source.index("cudaMalloc"))
        self.assertIn("cudaEventSynchronize(published)", source)
        header = (ROOT / "src/attention/qsa_sidecar_owner.h").read_text()
        self.assertIn("const std::uint8_t* payload() const noexcept", header)

    def test_identity_is_fixed_to_complete_payload_and_layer3_extent(self):
        header = (ROOT / "src/attention/qsa_sidecar_owner.h").read_text()
        self.assertIn("39'321'600", header)
        self.assertIn("3'276'800", header)
        self.assertIn("bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd", header)
        self.assertIn("payload_sha256", header)
        self.assertIn("layer3_sha256", header)

    def test_auth_cli_consumes_the_entire_rank_argument(self):
        source = (ROOT / "tests/qsa_sidecar_auth.cc").read_text()
        self.assertNotIn("std::stoi", source)
        self.assertIn('rank_argument != "0" && rank_argument != "1"', source)


if __name__ == "__main__":
    unittest.main()
