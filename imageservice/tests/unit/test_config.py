"""
tests/unit/test_config.py — Unit tests for startracker.config
"""

import dataclasses
import json
import os

import pytest

from workers.camera import CameraMode, RoiDefinition
from workers.config import ConfigError, SessionConfig, SystemConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SystemConfig loading
# ---------------------------------------------------------------------------

class TestSystemConfigLoading:

    def test_loads_valid_config(self, system_config):
        assert system_config.camera_serial_number == "TEST000001"
        assert system_config.sensor_width  == 4504
        assert system_config.sensor_height == 4504

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            SystemConfig.from_file(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            SystemConfig.from_file(str(p))

    def test_missing_serial_number_raises(self, system_config_dict, tmp_path):
        del system_config_dict["camera"]["serial_number"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        with pytest.raises(ConfigError, match="serial_number"):
            SystemConfig.from_file(str(p))

    def test_missing_sensor_section_raises(self, system_config_dict, tmp_path):
        del system_config_dict["sensor"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        with pytest.raises(ConfigError, match="sensor"):
            SystemConfig.from_file(str(p))

    def test_missing_paths_section_raises(self, system_config_dict, tmp_path):
        del system_config_dict["paths"]
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        with pytest.raises(ConfigError, match="paths"):
            SystemConfig.from_file(str(p))

    def test_wrong_bit_depth_raises(self, system_config_dict, tmp_path):
        system_config_dict["camera"]["output_bit_depth"] = 8
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        with pytest.raises(ConfigError, match="output_bit_depth"):
            SystemConfig.from_file(str(p))

    def test_deflate_level_out_of_range_raises(self, system_config_dict,
                                                tmp_path):
        system_config_dict["compression"]["deflate_level"] = 11
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        with pytest.raises(ConfigError, match="deflate_level"):
            SystemConfig.from_file(str(p))

    def test_defaults_applied(self, system_config):
        assert system_config.camera_transport_buffer_size == 8
        assert system_config.camera_output_bit_depth      == 12
        assert system_config.centroid_min_total_intensity == 1.0
        assert system_config.serial_port is None
        assert system_config.serial_baud_rate == 115200

    def test_extra_keys_ignored(self, system_config_dict, tmp_path):
        system_config_dict["unknown_section"] = {"foo": "bar"}
        system_config_dict["camera"]["unknown_key"] = 42
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(system_config_dict))
        # Should not raise
        cfg = SystemConfig.from_file(str(p))
        assert cfg is not None


# ---------------------------------------------------------------------------
# SystemConfig.save() — atomic write with backup
# ---------------------------------------------------------------------------

class TestSystemConfigSave:

    def test_save_and_reload(self, system_config, tmp_path):
        out = tmp_path / "saved.json"
        system_config.save(str(out))
        reloaded = SystemConfig.from_file(str(out))
        assert reloaded.camera_serial_number == system_config.camera_serial_number
        assert reloaded.sensor_width         == system_config.sensor_width

    def test_save_creates_backup(self, system_config, tmp_path):
        out = tmp_path / "config.json"
        system_config.save(str(out))
        # Save again — original should be backed up
        new_cfg = dataclasses.replace(
            system_config, camera_serial_number="NEWSN"
        )
        new_cfg.save(str(out))
        assert os.path.exists(str(out) + ".bak")

    def test_save_backup_contains_original(self, system_config, tmp_path):
        out = tmp_path / "config.json"
        system_config.save(str(out))
        new_cfg = dataclasses.replace(
            system_config, camera_serial_number="NEWSN"
        )
        new_cfg.save(str(out))
        bak = json.loads(open(str(out) + ".bak").read())
        assert bak["camera"]["serial_number"] == "TEST000001"

    def test_save_original_untouched_on_invalid(self, system_config, tmp_path,
                                                  monkeypatch):
        """If save validation fails, original file must be untouched."""
        out = tmp_path / "config.json"
        system_config.save(str(out))
        original_content = open(str(out)).read()

        # Patch from_file to always fail validation after write
        def bad_reload(path):
            raise ConfigError("forced validation failure")
        monkeypatch.setattr(SystemConfig, "from_file", staticmethod(bad_reload))

        with pytest.raises(ConfigError, match="Validation"):
            system_config.save(str(out))

        # Original must be intact
        assert open(str(out)).read() == original_content


# ---------------------------------------------------------------------------
# SystemConfig.to_camera_config()
# ---------------------------------------------------------------------------

class TestToCameraConfig:

    def test_single_roi_produces_correct_mode(self, system_config,
                                               session_config_single):
        cam_cfg = system_config.to_camera_config(session_config_single)
        assert cam_cfg.mode is CameraMode.SINGLE_ROI

    def test_single_roi_uses_session_exposure(self, system_config,
                                               session_config_single):
        cam_cfg = system_config.to_camera_config(session_config_single)
        assert cam_cfg.exposure_us == session_config_single.exposure_us

    def test_full_frame_produces_correct_mode(self, system_config,
                                               session_config_full_frame):
        cam_cfg = system_config.to_camera_config(session_config_full_frame)
        assert cam_cfg.mode is CameraMode.FULL_FRAME

    def test_serial_number_from_system_config(self, system_config,
                                               session_config_single):
        cam_cfg = system_config.to_camera_config(session_config_single)
        assert cam_cfg.serial_number == system_config.camera_serial_number


# ---------------------------------------------------------------------------
# SessionConfig validation
# ---------------------------------------------------------------------------

class TestSessionConfig:

    def test_valid_session(self, session_config_single):
        assert session_config_single.frame_rate_hz == 10.0
        assert session_config_single.n_frames == 20

    def test_zero_frame_rate_raises(self):
        with pytest.raises(ConfigError, match="frame_rate_hz"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=None,
                frame_rate_hz=0.0, exposure_us=1_000,
                n_frames=10, duration_s=None,
                session_id="test",
            )

    def test_exposure_exceeds_frame_period_raises(self):
        with pytest.raises(ConfigError, match="frame period"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=None,
                frame_rate_hz=100.0, exposure_us=20_000,  # > 10ms
                n_frames=10, duration_s=None,
                session_id="test",
            )

    def test_negative_n_frames_raises(self):
        with pytest.raises(ConfigError, match="n_frames"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=None,
                frame_rate_hz=1.0, exposure_us=1_000,
                n_frames=-1, duration_s=None,
                session_id="test",
            )

    def test_negative_duration_raises(self):
        with pytest.raises(ConfigError, match="duration_s"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=None,
                frame_rate_hz=1.0, exposure_us=1_000,
                n_frames=None, duration_s=-5.0,
                session_id="test",
            )

    def test_empty_session_id_raises(self):
        with pytest.raises(ConfigError, match="session_id"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=None,
                frame_rate_hz=1.0, exposure_us=1_000,
                n_frames=10, duration_s=None,
                session_id="",
            )

    def test_full_frame_rejects_rois(self):
        roi = RoiDefinition(128, 0, 128, 0, "roi", True)
        with pytest.raises(ConfigError, match="FULL_FRAME"):
            SessionConfig(
                mode=CameraMode.FULL_FRAME, rois=[roi],
                frame_rate_hz=1.0, exposure_us=1_000,
                n_frames=10, duration_s=None,
                session_id="test",
            )

    def test_single_roi_requires_one_roi(self):
        with pytest.raises(ConfigError, match="SINGLE_ROI"):
            SessionConfig(
                mode=CameraMode.SINGLE_ROI, rois=None,
                frame_rate_hz=10.0, exposure_us=1_000,
                n_frames=10, duration_s=None,
                session_id="test",
            )

    def test_is_bounded_both_set(self):
        s = SessionConfig(
            mode=CameraMode.FULL_FRAME, rois=None,
            frame_rate_hz=1.0, exposure_us=1_000,
            n_frames=10, duration_s=30.0,
            session_id="test",
        )
        assert s.is_bounded
        assert s.has_frame_limit
        assert s.has_time_limit

    def test_is_bounded_neither_set(self):
        s = SessionConfig(
            mode=CameraMode.FULL_FRAME, rois=None,
            frame_rate_hz=1.0, exposure_us=1_000,
            n_frames=None, duration_s=None,
            session_id="test",
        )
        assert not s.is_bounded

    def test_write_example_produces_valid_file(self, tmp_path,
                                                tmp_data_dir, tmp_log_dir):
        p = tmp_path / "example.json"
        SystemConfig.write_example(str(p))
        assert p.exists()
        # The example uses placeholder paths; patch them and reload
        raw = json.loads(p.read_text())
        raw["paths"]["data_dir"] = tmp_data_dir
        raw["paths"]["log_dir"]  = tmp_log_dir
        raw["camera"]["serial_number"] = "TEST"
        p2 = tmp_path / "example2.json"
        p2.write_text(json.dumps(raw))
        cfg = SystemConfig.from_file(str(p2))
        assert cfg is not None
