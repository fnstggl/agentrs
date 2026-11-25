"""Media tooling for creators and encoders (crucible.semantic.media): rendering, image processing, audio synthesis,
video assembly, transcoding, probing and decoding, with bounded resources. No neural models are used here."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

from crucible.semantic import media as M


def _png_size(data: bytes) -> tuple[int, int]:
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    return im.size


def test_render_text_wraps_and_differs_by_text():
    a = M.render_text_image("A short headline about the city bus network and three new routes", width=320)
    b = M.render_text_image("Something else entirely: a recipe for sponge cake", width=320)
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and a != b
    w, h = _png_size(a)
    assert w == 320 and h > 100          # wrapped onto several lines
    fixed = M.render_text_image("x " * 2000, width=200, height=150)
    assert _png_size(fixed) == (200, 150)  # a fixed canvas clips instead of growing


def test_render_html_lite_lays_out_structure_and_ignores_scripts():
    html = ("<html><head><title>Shop</title><style>p{color:red}</style><script>alert(1)</script></head>"
            "<body style='background:#202020;color:#ffffff'><h1>Big sale</h1><p>All <b>shoes</b> half price</p>"
            "<ul><li>red</li><li>blue</li></ul><pre>  code</pre><img alt='logo'></body></html>")
    title, text = M.html_text(html)
    assert title == "Shop"
    assert "Big sale" in text and "shoes" in text and "- red" in text and "alert" not in text
    img = M.open_image(M.render_html_image(html, width=400))
    px = np.asarray(img)
    assert img.width == 400
    assert tuple(px[2, 2]) == (32, 32, 32)                      # the body background colour
    assert (px > 200).all(axis=2).sum() > 200                  # white text was drawn


def test_draw_scene_and_process_image_ops():
    scene = M.draw_scene({"background": "#ffffff", "shapes": [{"kind": "circle", "xy": [0.25, 0.25, 0.75, 0.75],
                                                                "fill": "#ff0000"}]}, width=200, height=100)
    px = np.asarray(M.open_image(scene))
    assert tuple(px[50, 100]) == (255, 0, 0) and tuple(px[2, 2]) == (255, 255, 255)
    out = M.process_image(scene, [{"op": "scale", "f": 0.5}, {"op": "grayscale"},
                                  {"op": "encode", "fmt": "JPEG", "codec_quality": 50}])
    assert out[:2] == b"\xff\xd8"
    im = M.open_image(out)
    assert im.size == (100, 50)
    a = np.asarray(im).astype(int)
    assert np.abs(a[..., 0] - a[..., 1]).max() <= 3             # grey
    with pytest.raises(M.MediaError):
        M.process_image(scene, [{"op": "explode"}])


def test_open_image_refuses_garbage_and_pixel_bombs():
    with pytest.raises(M.MediaError):
        M.open_image(b"not an image at all")
    from PIL import Image
    big = Image.new("1", (8000, 8000))                        # 64 MP over the 40 MP limit, tiny as a 1-bit PNG
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    with pytest.raises(M.MediaError):
        M.open_image(buf.getvalue())


def test_speech_synthesis_and_audio_decoding():
    wav = M.synth_speech("Hello from the simulated internet.", voice="kal")
    info = M.probe(wav)
    assert info["has_audio"] and not info["has_video"] and 1.0 < info["duration_s"] < 6.0
    x = M.decode_audio(wav, 16000)
    assert x.dtype == np.float32 and abs(len(x) / 16000 - info["duration_s"]) < 0.1
    assert float(np.sqrt(np.mean(x ** 2))) > 0.01             # it is not silence
    with pytest.raises(M.MediaError):
        M.synth_speech("hi", voice="no_such_voice")


def test_synth_signals_are_what_they_claim():
    sr = 16000
    tone = M.synth_tone(440.0, 1.0, sr=sr)
    spec = np.abs(np.fft.rfft(tone))
    assert abs(np.argmax(spec) * sr / len(tone) - 440.0) < 2.0
    mel = M.synth_melody(["A4", None, ["C4", "E4"]], bpm=120, sr=sr)
    assert len(mel) == 3 * sr // 2 and np.abs(mel[sr // 2:sr]).max() == 0.0   # the rest is silent
    pink = M.synth_noise(2.0, color="pink", sr=sr, seed=1)
    p = np.abs(np.fft.rfft(pink)) ** 2
    f = np.fft.rfftfreq(len(pink), 1 / sr)
    assert p[(f > 50) & (f < 200)].mean() > 5 * p[(f > 4000) & (f < 6000)].mean()   # 1/f: more low-frequency power
    assert M.note_freq("A5") == pytest.approx(880.0)


def test_assemble_video_with_audio_then_transcode_trim_and_extract():
    imgs = [M.render_text_image(t, width=320, height=240, font_size=40) for t in ("one", "two", "three")]
    speech = M.synth_speech("one two three", voice="slt")
    vid = M.assemble_video(imgs, speech, seconds_per_image=1.0, size=(320, 240), fade_s=0.1)
    info = M.probe(vid)
    assert info["has_video"] and info["has_audio"] and (info["width"], info["height"]) == (320, 240)
    assert abs(info["duration_s"] - 3.0) < 0.2
    frames = M.extract_frames(vid, 3, info=info)
    assert len(frames) == 3 and frames[0].shape == (240, 320, 3)
    assert np.abs(frames[0].astype(int) - frames[2].astype(int)).mean() > 1.0   # different slides
    gray = M.gray_sequence(vid, fps=4.0, size=32)
    assert gray.shape[1:] == (32, 32) and 10 <= len(gray) <= 14
    small = M.transcode(vid, "mp4", width=160, crf=35, start_s=0.5)
    si = M.probe(small)
    assert si["width"] == 160 and abs(si["duration_s"] - 2.5) < 0.25
    webm = M.transcode(vid, "webm", crf=40)
    assert M.probe(webm)["format"].startswith("matroska,webm")
    mp3 = M.transcode(speech, "mp3", audio_bitrate="32k")
    assert M.probe(mp3)["format"] == "mp3"
    silent = M.assemble_video(imgs[:1], None, seconds_per_image=0.5, size=(64, 48))
    assert not M.probe(silent)["has_audio"]


def test_ffmpeg_refuses_playlists_and_other_files():
    """A crafted HLS playlist pointing at a local file must not be followed (demuxer allow-list)."""
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:///etc/hostname\n#EXT-X-ENDLIST\n"
    with pytest.raises(M.MediaError):
        M.decode_audio(playlist)
    concat = b"ffconcat version 1.0\nfile '/etc/hostname'\n"
    with pytest.raises(M.MediaError):
        M.probe(concat)


def test_tool_runs_are_time_and_size_bounded():
    lim = M.MediaLimits(timeout_s=0.5)
    work = M._Work({}, lim)
    try:
        with pytest.raises(M.MediaError, match="exceeded"):
            work.run([M.FFMPEG, "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                      "sine=frequency=440:duration=30", "-f", "null", "-"])
    finally:
        work.close()
    with pytest.raises(M.MediaError):
        M._Work({"in.bin": b"x" * 100}, M.MediaLimits(max_input_bytes=10))
    with pytest.raises(M.MediaError):
        M.assemble_video([M.render_text_image("x", width=64, height=64)], None, size=(4000, 4000))


def test_media_work_directories_are_removed():
    import tempfile
    base = tempfile.mkdtemp(prefix="crucible-media-test-")
    old = os.environ.get("CRUCIBLE_MEDIA_TMP")
    os.environ["CRUCIBLE_MEDIA_TMP"] = base
    try:
        M.probe(M.synth_speech("tidy", voice="slt"))
        assert os.listdir(base) == []
    finally:
        if old is None:
            os.environ.pop("CRUCIBLE_MEDIA_TMP", None)
        else:
            os.environ["CRUCIBLE_MEDIA_TMP"] = old
        os.rmdir(base)
