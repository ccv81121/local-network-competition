"""``prepare`` rebuilds the input files of the published runs byte for byte.

The digests are those of the protocol and space files the published spinel,
SP and DP searches read (numpy 2.4, pandas 3.0, scikit-learn 1.8).
"""
import hashlib
from pathlib import Path

import pytest

from symbolic_regression.inputs import prepare

TABLES = Path(__file__).resolve().parents[2] / "data" / "descriptor_tables"
A = "protocol/RF_PROTOCOL_BUNDLE/arrays/"
PUBLISHED = {
    "spinel": {
        "protocol/protocol_rows.csv": "66e19959bb43a05f3862517995cf1aa21477cd6bc52a3b7afce1ae6edbd673ab",
        A + "X_full_clean_float64.npy": "56ad8445ad3bd01201bd14f11bf511ea06b97a3c301228cf5adcc6aba5ec2e8d",
        A + "y_int8.npy": "67f8e8793316015d8730a3dd79bf1243e02ea5b7c84d0d350eb4a7d12a0e075a",
        A + "mask_all_primary_bool.npy": "b56e7ca81a284cced7b3f79a82050fab570154612c5bdad03eb4078d19e091f9",
        A + "mask_hard_core_bool.npy": "95c6c3f1e5ffa00953e14a7f94ffc2090c045ef1369e2c9033008e9e49fb15d0",
        A + "mask_soft_all_bool.npy": "81b3a6760718b2a89bf095e99c45230df191e8d5f1f8819a5d70e24d0d165e29",
        "protocol/RF_PROTOCOL_BUNDLE/features/formula_full_feature_names.json":
            "996fc48128094470b9b570885806e8b37b7a66a7bd26ea1bca94a41935be086c",
        "space/terminal_registry.csv": "2a03fce3e9ae7a604f6f0b30920200edbbf56690b2f770441d3b22a8cfca2238",
        "space/PHYSICAL_CHANNEL_ROLE_REGISTRY.csv":
            "e5abde1293e95faa7a41863adc0c889703759a2d5733c4ab22de99d1f8943026",
        "space/RF_SEARCH_GUIDANCE_REGISTRY.csv":
            "2582136658ed75c862ebe78e7b7b454842d129886359dc45fefd5634d53c72fe",
    },
    "sp": {
        "protocol/protocol_rows.csv": "c24da30ce0803cde2190296791dea7e19bd2ca544ce792a67eaa6ee0da3d98e6",
        A + "X_full_clean_float64.npy": "177ec75b7f75126df8193f3421d540745dabb6cac7a75e6c8d4c2b8465bdb678",
        A + "y_int8.npy": "00a6ee7263ac3f8389d8186f6b7ba510815c8b57ff5b1722471f8e5f73f69d7b",
        A + "mask_all_primary_bool.npy": "fb709c5896e1ab4210b50d7210d6854590119549a44eb158f6fd55c44d6580ed",
        A + "mask_hard_core_bool.npy": "fb709c5896e1ab4210b50d7210d6854590119549a44eb158f6fd55c44d6580ed",
        A + "mask_soft_all_bool.npy": "7f37669f7a0a9edc2dbd7a9bbd23fee6ea5821c06617af226e5ea16b32bd9fb0",
        A + "fold_id_int16.npy": "cb0bdb7714e63e6a89156ac9e68246036ed439bdfbfcfa8c8fb48c85a587ed7f",
        A + "mask_multiconfig_bool.npy": "9a86c9b48a89de28cc5424ac44fff2749e0d42845b8ce41abab4e2eb71399429",
        "protocol/RF_PROTOCOL_BUNDLE/features/formula_full_feature_names.json":
            "92f5ce51b376117d479e30c1be8c673dde03e8187ec73b9cd29f2eb011c122c3",
        "space/terminal_registry.csv": "e6a8b79f2d9a5594c54aa874162709d5a4559af49a2f48eb43cf196c9d4c78bb",
        "space/PHYSICAL_CHANNEL_ROLE_REGISTRY.csv":
            "61070da4f6710be25e478f8753eeb5c2c4af6133904c3f1f8d9224289cd0bed4",
        "space/RF_SEARCH_GUIDANCE_REGISTRY.csv":
            "bd7bc3a8d50da567fa1b09aebe399dc3bdc8dfd20eab38e4da686ded83ccb2b0",
    },
    "dp": {
        "protocol/protocol_rows.csv": "7c707a87dda49db3102dd9844f696987afdf1ea881244d70e5a1e8d604c9b069",
        A + "X_full_clean_float64.npy": "00eed032c11588277ddb5f0fd0446b6370025c5fd8f63da3dde24fa71d81f439",
        A + "y_int8.npy": "e1cb2bdc4b131532a36030031a03a235570519a2c04fb1350cf4e39857a79bca",
        A + "mask_all_primary_bool.npy": "2390f83628cd8e4a4315d6d6ec618746bb92de11eaa8356cb83c92efb3f6b13d",
        A + "mask_hard_core_bool.npy": "2390f83628cd8e4a4315d6d6ec618746bb92de11eaa8356cb83c92efb3f6b13d",
        A + "mask_soft_all_bool.npy": "4bddbdbb54418085b09a13aa69980239b1d570bff26f82969bae16686538e27b",
        A + "fold_id_int16.npy": "7428df04bb4eed275a27de5853a8a7475f1215176d0084405942d1b28defe349",
        A + "mask_multiconfig_bool.npy": "4bddbdbb54418085b09a13aa69980239b1d570bff26f82969bae16686538e27b",
        "protocol/RF_PROTOCOL_BUNDLE/features/formula_full_feature_names.json":
            "ae7617cb78e594a5a4e4a39260cb13331029c35e7b8e3f74dea9bba2e11db817",
        "space/terminal_registry.csv": "3849a29e06feffc2e4de6ab37ddcd665fcbe7b120bc8141de6245ee499516a8a",
        "space/PHYSICAL_CHANNEL_ROLE_REGISTRY.csv":
            "fd3a6451a74717fbdd4c5da8b29fda47ce97ed32559c2cd97f325bf5af59bff5",
        "space/RF_SEARCH_GUIDANCE_REGISTRY.csv":
            "084fdd679ba00c24a5704d1538632f52851b4fe93deb5686d121b5f5df00282b",
    },
    # inputs of the spinel two-edge search (681 retained states)
    "spinel-edges": {
        "protocol/arrays/X_full_clean_float64.npy":
            "df9e1defa2b992a9508551319b51c3e3e39608964e8aa77cd7677598ce2715d0",
        "protocol/arrays/sample_id_unicode.npy":
            "43e6739e049d308383b45185d15d4666f47f935a70ae6efd7d55e66f0d02e265",
        "protocol/features/formula_full_feature_names.json":
            "996fc48128094470b9b570885806e8b37b7a66a7bd26ea1bca94a41935be086c",
    },
}


@pytest.mark.skipif(not TABLES.is_dir(), reason="data/descriptor_tables not present")
@pytest.mark.parametrize("family", ["sp", "dp", "spinel", "spinel-edges"])
def test_prepare_reproduces_published_inputs(family, tmp_path):
    out = tmp_path / family
    prepare(family, TABLES / family.replace("-", "_"), out)
    digests = {rel: hashlib.sha256((out / rel).read_bytes()).hexdigest()
               for rel in PUBLISHED[family]}
    assert digests == PUBLISHED[family]
