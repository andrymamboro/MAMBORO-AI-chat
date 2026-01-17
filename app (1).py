import gradio as gr
import torch
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# 1. Konfigurasi Model
model_id = "cognitivecomputations/dolphin-2.9-llama3-8b"
print(f"Sedang memuat model {model_id}...")

# 2. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Fungsi Chat
@spaces.GPU
def chat_response(message, history):
    # --- BAGIAN PENTING: SYSTEM PROMPT BAHASA INDONESIA ---
    # Kita memberi "mantra" agar model patuh berbahasa Indonesia
    system_prompt = "Kamu adalah asisten AI yang cerdas dan membantu. Kamu WAJIB menjawab setiap pertanyaan pengguna menggunakan Bahasa Indonesia yang baik dan jelas."
    
    # Format ChatML (System -> User -> Assistant)
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    for user_msg, bot_msg in history:
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    except Exception as e:
        yield f"Error Tokenizing: {str(e)}"
        return

    # Setup Streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Konfigurasi Generasi (Tuned for Indonesian)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=2048, 
        temperature=0.6,       # Sedikit diturunkan agar lebih fokus
        repetition_penalty=1.15, # PENTING: Mencegah kata berulang (looping) di Bhs Indo
        do_sample=True
    )

    # Wrapper Thread
    def run_generation():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            print(f"Error generation: {e}")
        finally:
            streamer.end()

    # Jalankan Thread
    thread = Thread(target=run_generation)
    thread.start()

    # Streaming Output
    response = ""
    try:
        for new_text in streamer:
            response += new_text
            yield response
    except Exception as e:
        yield response + f"\n\n[Sistem Error: {str(e)}]"

# 4. Tampilan Web
judul_html = """
    <div style="display: flex; align-items: center;">
        <img src="https://huggingface.co/spaces/obitouchiha88/jarvis_tes/resolve/main/logo.png" width="40" height="40" style="margin-right: 10px; border-radius: 5px;">
        <span>MAMBORO-AI Chat</span>
    </div>
"""


demo = gr.ChatInterface(
    fn=chat_response,
    # Menggunakan HTML untuk menampilkan gambar
    # src='logo.png' artinya mengambil file lokal bernama logo.png
    title=judul_html,
    description="Model 🐬 Dolphin-2.9-llama3-8b yang dioptimalkan untuk Bahasa Indonesia. [Andri on Github](https://github.com/andrymamboro)",
    examples=["Apa makanan khas Suku Kaili?", "Buatkan surat lamaran kerja", "Jelaskan sejarah Majapahit"]
)

demo.launch(share=True,favicon_path="logo2.png")