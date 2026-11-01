"""GitHub'da çalışan anlatıcı TTS işçisi (Tale, 27 Ekim). Bu dosya AÇIK depoda durur: metin içermez, metni şifreli işten okur.
Çağrı: python tts_isci.py <grup> <iş klasörü> <çıktı klasörü> <referans klasörü>
Çıktı: <id>.wav (Chatterbox ya da Whisper için verilen ses) + <id>.json ({"kelimeler": [[kelime, bas, bit]]}).
Hata olursa yalnız hata türü ve satır numaraları yazılır (metin günlüğe sızmasın)."""
import json
import os
import shutil
import sys
import traceback


def main():
    grup, isd, cikti, refd = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(cikti, exist_ok=True)
    with open(os.path.join(isd, "is.json"), encoding="utf-8") as f:
        is_ = json.load(f)
    ogeler = [x for x in is_["ogeler"] if x.get("grup") == grup]
    cb = [x for x in ogeler if x["tur"] == "chatterbox"]
    if cb:
        import torch
        import torchaudio as ta
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        ref = os.path.join(refd, is_["ses"] + ".wav")
        for x in cb:
            torch.manual_seed(int(x.get("seed") or 1))
            kw = {k: float(x[a]) for k, a in (("exaggeration", "exag"), ("cfg_weight", "cfg")) if x.get(a) is not None}
            try:
                wav = m.generate(x["metin"], language_id="tr", audio_prompt_path=ref, **kw)
            except TypeError:  # sürüm bu ayarları almıyorsa varsayılanla
                wav = m.generate(x["metin"], language_id="tr", audio_prompt_path=ref)
            ta.save(os.path.join(cikti, x["id"] + ".wav"), wav, m.sr)
        del m
    for x in ogeler:
        if x["tur"] == "whisper":
            shutil.copy(os.path.join(isd, "wav", x["id"] + ".wav"), os.path.join(cikti, x["id"] + ".wav"))
    from faster_whisper import WhisperModel
    wm = WhisperModel(is_.get("whisper") or "small", device="cpu", compute_type="int8")
    for x in ogeler:
        f = os.path.join(cikti, x["id"] + ".wav")
        if not os.path.exists(f):
            continue
        parca, _ = wm.transcribe(f, language="tr", beam_size=5, temperature=0, condition_on_previous_text=False,
                                 word_timestamps=True, vad_filter=False)
        kel = [[w.word.strip(), round(w.start, 3), round(w.end, 3)] for s in parca for w in (s.words or [])]
        with open(os.path.join(cikti, x["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump({"kelimeler": kel}, f, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        with open(os.path.join(sys.argv[3], "hata.txt"), "w") as f:
            f.write(type(e).__name__ + " · " + " | ".join(f"{os.path.basename(t.filename)}:{t.lineno}" for t in tb[-8:]))
        sys.exit(1)
