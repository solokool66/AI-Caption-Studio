import os
import re
import random
import tempfile
import subprocess
import sys
import wave
import gc
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
from faster_whisper import WhisperModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Support large files up to 50 GB (handles 2GB, 5GB, 10GB+ videos seamlessly)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024

MODELS = {}
CPU_THREADS = os.cpu_count() or 4

def get_model(size="base"):
    global MODELS
    size = size.lower().strip()
    if size not in ['tiny', 'base', 'small']:
        size = 'base'
    if size not in MODELS:
        print("==================================================")
        print(f"[*] Loading Faster-Whisper '{size}' model (threads: {CPU_THREADS}, int8)...")
        print("==================================================")
        MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
        print(f"[+] Whisper Model '{size}' loaded successfully!")
    return MODELS[size]

# Pre-warm the base model at startup for instant readiness
try:
    get_model('base')
except Exception as e:
    print(f"[!] Model initialization deferred: {e}")

# Predefined colors mapping Hex -> ASS BGR format (&HBBGGRR&)
COLOR_PRESETS = {
    'gold': '&H00E6FF&',      # Vibrant CapCut Yellow #FFE600
    'yellow': '&H00E6FF&',    # #FFE600
    'lime': '&H66FF00&',      # #00FF66
    'cyan': '&HFFE500&',      # #00E5FF
    'pink': '&H852AFF&',      # #FF2A85
    'purple': '&HF755A8&',    # #A855F7
    'coral': '&H4D4DFF&',     # #FF4D4D
    'white': '&HFFFFFF&',     # #FFFFFF
}

BG_CANVAS_MAP = {
    'green': ('0x00FF00', '#00FF00'),
    'blue': ('0x0000FF', '#0000FF'),
    'black': ('0x000000', '#000000'),
}

def hex_to_ass(hex_val, default='&H00AAFF&'):
    if not hex_val:
        return default
    hex_val = hex_val.strip().lower()
    if hex_val in COLOR_PRESETS:
        return COLOR_PRESETS[hex_val]
    hex_clean = hex_val.lstrip('#')
    if len(hex_clean) == 6:
        r, g, b = hex_clean[0:2], hex_clean[2:4], hex_clean[4:6]
        return f"&H{b}{g}{r}&"
    return default

# Clean word normalization (preserves genuine speech without forced word corruptions)
def correct_kjv_word(word):
    return word

def format_time_ass(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02}:{s:02}.{cs:02}"

def generate_ass_header(aspect_ratio='9:16', font_name='Arial Black'):
    """
    Generate ASS header configured for 9:16 Vertical (Reels/TikTok/CapCut)
    or 16:9 Landscape (YouTube).
    """
    if aspect_ratio == '9:16':
        res_x = 1080
        res_y = 1920
        font_size = 112
        margin_v = 640   # Positioned in lower-middle action zone
        margin_lr = 45
        outline = 6.0
        shadow = 0.0     # Clean crisp viral outline, eliminates blurry double-shadow
    else: # 16:9 Landscape
        res_x = 1920
        res_y = 1080
        font_size = 98
        margin_v = 140   # Above bottom scrubber
        margin_lr = 70
        outline = 5.0
        shadow = 0.0     # Clean crisp viral outline, eliminates blurry double-shadow

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header

def split_segment_into_chunks(segment, style='typing', highlight_ass='&H00AAFF&', active_scale=135, lead_offset=0.05):
    """
    Splits Whisper word timestamps into kinetic caption dialogue lines with:
    1. Calibrated vocal lookahead (lead_offset=0.05s) for natural acoustic sync.
    2. Intelligent sentence and punctuation chunking (never bleeds across .!? or long pauses).
    3. Clean decay buffer on sentence ends so words never awkwardly hang across silences.
    """
    if not hasattr(segment, 'words') or not segment.words:
        return []

    words = segment.words
    total_words = len(words)
    if total_words == 0:
        return []

    # 1. Break words into natural syntactic phrases based on punctuation and pauses
    phrases = []
    curr_phrase = []
    import re

    for i, w in enumerate(words):
        curr_phrase.append(w)
        text = w.word.strip() if hasattr(w, 'word') and w.word else ''
        is_sentence_end = bool(re.search(r'[.!?]$', text))
        
        has_long_pause = False
        if i < len(words) - 1:
            pause_dur = words[i+1].start - w.end
            if pause_dur >= 0.70:
                has_long_pause = True
                
        if is_sentence_end or has_long_pause:
            phrases.append(curr_phrase)
            curr_phrase = []
    if curr_phrase:
        phrases.append(curr_phrase)

    chunks = []
    pop_scale = active_scale + 15

    if style == 'punch':
        max_chunk = 2
    elif style == 'typing':
        max_chunk = 4
    else: # marker, bounce, glow
        max_chunk = 4

    for phrase in phrases:
        p_len = len(phrase)
        idx = 0
        while idx < p_len:
            rem = p_len - idx
            if rem <= max_chunk:
                c_size = rem
            elif rem == 5 and max_chunk >= 3:
                c_size = 3
            else:
                c_size = min(3, max_chunk)
                if idx + 2 < p_len and hasattr(phrase[idx + 1], 'word') and re.search(r'[,;:]$', phrase[idx + 1].word.strip()):
                    c_size = 2
                elif idx + 3 < p_len and hasattr(phrase[idx + 2], 'word') and re.search(r'[,;:]$', phrase[idx + 2].word.strip()):
                    c_size = 3

            chunk_words = phrase[idx : idx + c_size]

            for i, active_word in enumerate(chunk_words):
                raw_start = active_word.start
                raw_end = active_word.end

                # Apply lead offset for instant acoustic onset sync (eliminates perceived lag)
                line_start = max(0.0, raw_start - lead_offset)
                
                if i < len(chunk_words) - 1:
                    next_start = max(line_start + 0.12, chunk_words[i+1].start - lead_offset)
                    line_end = next_start
                else:
                    # Last word in chunk: hold through vocal decay
                    line_end = max(line_start + 0.20, raw_end + 0.12)

                if line_end <= line_start:
                    line_end = line_start + 0.15

                styled_words = []

                if style == 'marker':
                    # CapCut Marker Box: Active word has black text inside a solid highlight box
                    for j, w in enumerate(chunk_words):
                        word_text = correct_kjv_word(w.word.strip() if hasattr(w, 'word') and w.word else '')
                        if j == i:
                            pop_tag = f"{{\\c&H000000&\\3c{highlight_ass}\\bord12\\shad0\\blur0\\fscx112\\fscy112\\t(0,90,\\fscx108\\fscy108)}}"
                            reset_tag = "{\\c&HFFFFFF&\\3c&H000000&\\bord6\\shad0\\blur0\\fscx100\\fscy100}"
                            styled_words.append(f"{pop_tag}{word_text}{reset_tag}")
                        else:
                            styled_words.append(word_text)

                elif style == 'typing':
                    # Typewriter: Render words cumulatively from 0 up to active word i
                    for j in range(i + 1):
                        w = chunk_words[j]
                        word_text = correct_kjv_word(w.word.strip() if hasattr(w, 'word') and w.word else '')
                        if j == i:
                            pop_tag = f"{{\\c{highlight_ass}\\3c&H000000&\\bord6.5\\shad0\\blur0\\fscx{pop_scale}\\fscy{pop_scale}\\t(0,100,\\fscx{active_scale}\\fscy{active_scale})}}"
                            reset_tag = "{\\c&HFFFFFF&\\3c&H000000&\\bord6\\shad0\\blur0\\fscx100\\fscy100}"
                            styled_words.append(f"{pop_tag}{word_text}{reset_tag}")
                        else:
                            styled_words.append(word_text)

                elif style == 'glow':
                    for j, w in enumerate(chunk_words):
                        word_text = correct_kjv_word(w.word.strip() if hasattr(w, 'word') and w.word else '')
                        if j == i:
                            active_tag = f"{{\\c{highlight_ass}\\3c&H000000&\\bord7\\shad0\\blur0\\fscx{active_scale}\\fscy{active_scale}}}"
                            reset_tag = "{\\c&HFFFFFF&\\3c&H000000&\\bord6\\shad0\\blur0\\fscx100\\fscy100}"
                            styled_words.append(f"{active_tag}{word_text}{reset_tag}")
                        else:
                            styled_words.append(word_text)

                else: # 'bounce', 'yellow_text', 'punch'
                    for j, w in enumerate(chunk_words):
                        word_text = correct_kjv_word(w.word.strip() if hasattr(w, 'word') and w.word else '')
                        if j == i:
                            pop_tag = f"{{\\c{highlight_ass}\\3c&H000000&\\bord6.5\\shad0\\blur0\\fscx{pop_scale}\\fscy{pop_scale}\\t(0,120,\\fscx{active_scale}\\fscy{active_scale})}}"
                            reset_tag = "{\\c&HFFFFFF&\\3c&H000000&\\bord6\\shad0\\blur0\\fscx100\\fscy100}"
                            styled_words.append(f"{pop_tag}{word_text}{reset_tag}")
                        else:
                            styled_words.append(word_text)

                chunk_text = " ".join(styled_words)
                chunks.append((line_start, line_end, chunk_text))

            idx += c_size

    return chunks

class OffsetWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

class OffsetSegment:
    def __init__(self, start, end, words):
        self.start = start
        self.end = end
        self.words = words

def extract_wav_chunk(in_wav, out_wav, start_s, end_s):
    with wave.open(in_wav, 'rb') as wf_in:
        framerate = wf_in.getframerate()
        n_channels = wf_in.getnchannels()
        sampwidth = wf_in.getsampwidth()
        start_frame = int(start_s * framerate)
        end_frame = int(end_s * framerate)
        num_frames = max(0, end_frame - start_frame)
        wf_in.setpos(start_frame)
        data = wf_in.readframes(num_frames)
    with wave.open(out_wav, 'wb') as wf_out:
        wf_out.setnchannels(n_channels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(framerate)
        wf_out.writeframes(data)

def transcribe_audio_chunked(model, wav_path, chunk_target_s=480, beam_size=2):
    """
    Transcribes audio safely in memory-capped streaming chunks (default ~8 mins).
    Guarantees that 2GB - 10GB (1hr to 10hr+) files never exceed RAM or fail
    with 'Unable to allocate X GiB for an array with shape ... complex128'.
    """
    with wave.open(wav_path, 'rb') as wf:
        framerate = wf.getframerate()
        total_duration = wf.getnframes() / float(framerate)

    print(f"[*] Audio total duration: {total_duration:.1f}s ({total_duration/60:.1f} mins)")

    # For files under 10 minutes, single pass is fast and uses minimal RAM (<150MB)
    if total_duration <= chunk_target_s * 1.25:
        segments, _ = model.transcribe(
            wav_path,
            beam_size=beam_size,
            best_of=beam_size,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=True,
            word_timestamps=True,
            initial_prompt=None
        )
        for s in segments:
            yield s
        return

    # For multi-hour files (like Exodus 40, 3+ hours), calculate split points at natural speech pauses
    splits = [0.0]
    curr_target = chunk_target_s
    n_frames = int(total_duration * framerate)

    with wave.open(wav_path, 'rb') as wf:
        while curr_target < total_duration - (chunk_target_s * 0.25):
            win_start_s = max(splits[-1] + 10.0, curr_target - 15.0)
            win_end_s = min(total_duration - 2.0, curr_target + 15.0)

            if win_end_s <= win_start_s:
                best_split = curr_target
            else:
                start_frame = max(0, min(n_frames - 1, int(win_start_s * framerate)))
                end_frame = max(start_frame + 1, min(n_frames, int(win_end_s * framerate)))
                num_frames = end_frame - start_frame
                wf.setpos(start_frame)
                raw = wf.readframes(num_frames)
                audio = np.frombuffer(raw, dtype=np.int16)
                
                sub_len = int(0.25 * framerate)
                if len(audio) > sub_len:
                    step = int(0.05 * framerate)
                    energies = [np.mean(np.abs(audio[i:i+sub_len])) for i in range(0, len(audio) - sub_len, step)]
                    min_idx = np.argmin(energies)
                    best_split = win_start_s + (min_idx * step + sub_len // 2) / framerate
                else:
                    best_split = curr_target
                    
            splits.append(round(float(best_split), 2))
            curr_target = best_split + chunk_target_s
            
    splits.append(round(float(total_duration), 2))

    total_chunks = len(splits) - 1
    print(f"[*] Large audio partitioned into {total_chunks} safe memory chunks (8-min intervals)")

    temp_chunk_dir = os.path.join(os.path.dirname(wav_path), "chunks")
    os.makedirs(temp_chunk_dir, exist_ok=True)

    for idx in range(total_chunks):
        c_start = splits[idx]
        c_end = splits[idx + 1]
        chunk_file = os.path.join(temp_chunk_dir, f"chunk_{idx}.wav")
        extract_wav_chunk(wav_path, chunk_file, c_start, c_end)

        print(f"[*] Transcribing chunk {idx+1}/{total_chunks} ({c_start:.1f}s -> {c_end:.1f}s)...")
        segments, _ = model.transcribe(
            chunk_file,
            beam_size=beam_size,
            best_of=beam_size,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=True,
            word_timestamps=True,
            initial_prompt=None
        )

        for s in segments:
            adj_words = []
            if hasattr(s, 'words') and s.words:
                for w in s.words:
                    adj_words.append(OffsetWord(
                        word=w.word,
                        start=round(w.start + c_start, 3),
                        end=round(w.end + c_start, 3)
                    ))
            yield OffsetSegment(
                start=round(s.start + c_start, 3),
                end=round(s.end + c_start, 3),
                words=adj_words
            )

        try:
            os.remove(chunk_file)
        except Exception:
            pass
        gc.collect()

    try:
        os.rmdir(temp_chunk_dir)
    except Exception:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio or video file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # User style and speed options
    speed_mode = request.form.get('speed', 'base').lower().strip()
    style = request.form.get('style', 'typing').lower().strip()
    color_choice = request.form.get('color', 'gold')
    highlight_ass = hex_to_ass(color_choice)

    try:
        active_scale = int(request.form.get('scale', 135))
    except (ValueError, TypeError):
        active_scale = 135

    active_scale = max(110, min(active_scale, 170))
    aspect_ratio = request.form.get('aspect_ratio', '9:16')
    if aspect_ratio not in ['9:16', '16:9']:
        aspect_ratio = '9:16'

    bg_choice = request.form.get('bg_color', 'green').lower()
    canvas_color, _ = BG_CANVAS_MAP.get(bg_choice, ('0x00FF00', '#00FF00'))

    font_choice = request.form.get('font', 'Arial Black')
    allowed_fonts = {
        'Arial Black': 'Arial Black',
        'Impact': 'Impact',
        'Georgia': 'Georgia',
        'Montserrat': 'Montserrat',
        'Trebuchet MS': 'Trebuchet MS',
        'Outfit': 'Outfit',
        'Bangers': 'Bangers',
        'Courier Prime': 'Courier New'
    }
    font_name = allowed_fonts.get(font_choice, 'Arial Black')

    if file:
        temp_dir = tempfile.mkdtemp()
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_media_path = os.path.join(temp_dir, f"input_media{ext}")

        print(f"[*] Saving uploaded file ({file.filename}) to temporary directory...")
        file.save(input_media_path)

        extracted_wav = os.path.join(temp_dir, "audio_16k.wav")
        clean_audio_aac = os.path.join(temp_dir, "audio_clean.aac")
        output_ass = os.path.join(temp_dir, "captions.ass")
        output_mp4 = os.path.join(temp_dir, "overlay.mp4")

        try:
            # 1. High-speed Audio Extraction: Extract 16kHz mono WAV for Whisper
            # Community Edition: Capped to first 60 seconds (-t 60) for free preview/evaluation
            MAX_COMMUNITY_DURATION = 60
            print(f"[*] [Community Edition] Extracting first {MAX_COMMUNITY_DURATION}s of audio from {file.filename} using FFmpeg...")
            cmd_extract_wav = [
                "ffmpeg", "-y",
                "-i", input_media_path,
                "-t", str(MAX_COMMUNITY_DURATION),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                extracted_wav
            ]
            subprocess.run(cmd_extract_wav, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Extract clean AAC audio track for the final green-screen overlay (capped to 60s)
            print(f"[*] [Community Edition] Extracting AAC audio track (max {MAX_COMMUNITY_DURATION}s) for overlay multiplexing...")
            cmd_extract_aac = [
                "ffmpeg", "-y",
                "-i", input_media_path,
                "-t", str(MAX_COMMUNITY_DURATION),
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                clean_audio_aac
            ]
            subprocess.run(cmd_extract_aac, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. Transcribe with Whisper (Chunked streaming for multi-hour 2GB-10GB support)
            print(f"[*] Transcribing audio with Faster-Whisper '{speed_mode}' (CPU threads: {CPU_THREADS}, chunked streaming)...")
            whisper_model = get_model(speed_mode)
            beam = 3 if speed_mode == 'small' else 2
            segments = transcribe_audio_chunked(whisper_model, extracted_wav, chunk_target_s=480, beam_size=beam)

            print(f"[*] Generating kinetic ASS typography ({output_ass})...")
            ass_header = generate_ass_header(aspect_ratio=aspect_ratio, font_name=font_name)

            raw_dialogues = []
            for segment in segments:
                chunks = split_segment_into_chunks(
                    segment,
                    style=style,
                    highlight_ass=highlight_ass,
                    active_scale=active_scale
                )
                for c in chunks:
                    raw_dialogues.append(c)

            # Reconcile timestamps so no two dialogue events overlap in time.
            # Eliminates libass subtitle collisions where the previous text renders as a ghost/shadow behind the active text.
            clean_dialogues = []
            for idx in range(len(raw_dialogues)):
                start_s, end_s, text = raw_dialogues[idx]
                if clean_dialogues and start_s < clean_dialogues[-1][1]:
                    prev_s, prev_e, prev_t = clean_dialogues[-1]
                    clean_dialogues[-1] = (prev_s, start_s, prev_t)
                clean_dialogues.append((start_s, end_s, text))

            with open(output_ass, "w", encoding="utf-8-sig") as f:
                f.write(ass_header)
                for (start_s, end_s, chunk_text) in clean_dialogues:
                    start = format_time_ass(start_s)
                    end = format_time_ass(end_s)
                    f.write(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{chunk_text}\n")

            # Determine video canvas resolution
            if aspect_ratio == '9:16':
                canvas_res = "1080x1920"
            else:
                canvas_res = "1920x1080"

            print(f"[*] Rendering chroma video overlay with FFmpeg ({canvas_res}, {canvas_color}, 30fps)...")
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c={canvas_color}:s={canvas_res}:r=30",
                "-i", clean_audio_aac,
                "-vf", "ass=captions.ass",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-threads", "0",
                "-tune", "fastdecode",
                "-c:a", "copy",
                "-shortest",
                "overlay.mp4"
            ]
            subprocess.run(cmd, cwd=temp_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            base_name = os.path.splitext(file.filename)[0]
            # Normalize en-dash, em-dash, and non-ASCII characters to safe ASCII for HTTP headers & filenames
            base_name_safe = base_name.replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', "'").replace('\u2019', "'")
            base_name_ascii = re.sub(r'[^\x00-\x7F]+', '_', base_name_safe)
            download_name = f"{base_name_ascii}_{style}_{aspect_ratio.replace(':', 'x')}_overlay.mp4"

            # Also save a permanent copy into C:\MY DEV\app\output\
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            permanent_copy = os.path.join(output_dir, download_name)
            try:
                import shutil
                shutil.copy2(output_mp4, permanent_copy)
                print(f"[+] Saved permanent copy to: {permanent_copy}")
            except Exception as copy_err:
                print(f"[!] Could not copy to output folder: {copy_err}")

            response = send_file(
                output_mp4,
                as_attachment=True,
                mimetype='video/mp4',
                download_name=download_name
            )
            # Ensure custom HTTP header is strictly ASCII to prevent latin-1 encoding errors in WSGI
            response.headers['X-Output-Path'] = permanent_copy.encode('ascii', 'ignore').decode('ascii')
            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[!] Error during processing: {str(e)}", flush=True)
            return jsonify({'error': str(e)}), 500

@app.route('/open-output', methods=['GET'])
def open_output_folder():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    try:
        os.startfile(output_dir)
        return jsonify({'status': 'opened', 'path': output_dir})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5200))
    print(f"[*] Starting AI Caption Studio (Community Edition) on http://127.0.0.1:{port} ...")
    app.run(host='127.0.0.1', port=port, debug=False)
