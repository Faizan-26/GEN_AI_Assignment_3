import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Assignment 3 — GAN Showcase",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Cards */
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
  }

  /* Gradient header */
  .gradient-text {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem;
  }

  /* Section headers */
  .section-header {
    color: #a78bfa;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid rgba(167,139,250,0.3);
    padding-bottom: 0.3rem;
  }

  /* Badge */
  .badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }

  /* Metric boxes */
  .metric-box {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
  }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
  .metric-label { font-size: 0.78rem; color: #9ca3af; margin-top: 2px; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.4);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
  }

  img { border-radius: 10px; }

  .stAlert { border-radius: 10px; }

  /* Sidebar radio buttons */
  [data-testid="stSidebar"] .stRadio > div { gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ─── Model Path Resolution ────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def model_path(relative_path: str) -> str:
    return os.path.join(APP_DIR, relative_path)

MODEL_PATHS = {
    "dcgan":     model_path("question_1_model/dcgan_generator_best.pth"),
    "wgan":      model_path("question_1_model/wgan-gp_generator_best.pth"),
    "pix2pix_g": model_path("question_2_model/pix2pix_generator_final.pth"),
    "cyclegan": model_path("question_3_model/cyclegan_weights.pt"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definitions ────────────────────────────────────────────────────────

# ── Q1: DCGAN Generator ──
class DCGANGenerator(nn.Module):
    def __init__(self, z_input=100, feature_map=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_input, feature_map * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map * 4, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map * 2, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1),
            nn.BatchNorm2d(feature_map, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ── Q1: WGAN-GP Generator  (same architecture) ──
class WGANGenerator(nn.Module):
    def __init__(self, z_input=100, feature_map=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_input, feature_map * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ── Q2: Pix2Pix U-Net Generator ──
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2) if down else nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.bn(self.act(self.conv(x))))


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, f=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, f, 4, 2, 1), nn.LeakyReLU(0.2))
        self.e2 = UNetBlock(f, f * 2, down=True)
        self.e3 = UNetBlock(f * 2, f * 4, down=True)
        self.e4 = UNetBlock(f * 4, f * 8, down=True)
        self.e5 = UNetBlock(f * 8, f * 8, down=True)
        self.e6 = UNetBlock(f * 8, f * 8, down=True)
        self.e7 = UNetBlock(f * 8, f * 8, down=True)
        self.e8 = UNetBlock(f * 8, f * 8, down=True, use_bn=False)

        self.d1 = UNetBlock(f * 8, f * 8, down=False, dropout=True)
        self.d2 = UNetBlock(f * 16, f * 8, down=False, dropout=True)
        self.d3 = UNetBlock(f * 16, f * 8, down=False, dropout=True)
        self.d4 = UNetBlock(f * 16, f * 8, down=False)
        self.d5 = UNetBlock(f * 16, f * 4, down=False)
        self.d6 = UNetBlock(f * 8, f * 2, down=False)
        self.d7 = UNetBlock(f * 4, f, down=False)
        self.d8 = nn.Sequential(nn.ConvTranspose2d(f * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.d8(torch.cat([d7, e1], 1))


# ── Q3: CycleGAN Generator (ResNetGenerator — matches updated notebook) ──
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)  # residual connection


class ResNetGenerator(nn.Module):
    """
    ResNet-based generator for CycleGAN.
    Input/Output: (B, 3, 128, 128)
    Matches the architecture saved in cyclegan_weights.pt.
    """
    def __init__(self, in_ch=3, out_ch=3, f=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, f, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(f), nn.ReLU(True),
            # Downsample x2
            nn.Conv2d(f,   f*2, 3, 2, 1, bias=False), nn.InstanceNorm2d(f*2), nn.ReLU(True),
            nn.Conv2d(f*2, f*4, 3, 2, 1, bias=False), nn.InstanceNorm2d(f*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers.append(ResNetBlock(f * 4))
        layers += [
            # Upsample x2
            nn.ConvTranspose2d(f*4, f*2, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f*2), nn.ReLU(True),
            nn.ConvTranspose2d(f*2, f,   3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f),   nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(f, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Model Loading (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_dcgan():
    path = MODEL_PATHS["dcgan"]
    if not os.path.exists(path):
        return None
    model = DCGANGenerator(z_input=100, feature_map=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


@st.cache_resource
def load_wgan():
    path = MODEL_PATHS["wgan"]
    if not os.path.exists(path):
        return None
    model = WGANGenerator(z_input=100, feature_map=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


@st.cache_resource
def load_pix2pix():
    path = MODEL_PATHS["pix2pix_g"]
    if not os.path.exists(path):
        return None
    model = UNetGenerator()
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Strip DataParallel prefix if present
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state)
    model.eval()
    return model.to(DEVICE)


@st.cache_resource
def load_cyclegan():
    path = MODEL_PATHS["cyclegan"]
    if not os.path.exists(path):
        return None, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    def _build_and_load(state_dict):
        model = ResNetGenerator()
        # Remap 'module.model.' prefix to 'net.' (from DataParallel-wrapped checkpoint)
        clean = {k.replace("module.model.", "net."): v for k, v in state_dict.items()}
        model.load_state_dict(clean)
        model.eval()
        return model.to(DEVICE)
    g_ab = _build_and_load(checkpoint["G_AB"])
    g_ba = _build_and_load(checkpoint["G_BA"])
    return g_ab, g_ba


# ─── Utility Helpers ─────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (C,H,W) tensor in [-1,1] to a PIL Image."""
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1) / 2  # → [0,1]
    t = t.permute(1, 2, 0).numpy()
    t = (t * 255).astype(np.uint8)
    return Image.fromarray(t)


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """Resize a PIL image and return a (1,C,H,W) tensor in [-1,1]."""
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def img_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def show_model_status(name: str, loaded: bool):
    if loaded:
        st.markdown(f'<span class="badge">✅ {name}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="display:inline-block;background:rgba(239,68,68,0.2);border:1px solid rgba(239,68,68,0.4);color:#fca5a5;padding:2px 12px;border-radius:20px;font-size:0.78rem;font-weight:600;margin:2px">❌ {name} — model not found</span>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎨 GenAI Assignment 3")
    # st.markdown("*22F-3858 Faizan Tariq & 22F-3060 Hassaan Ch*")
    st.divider()

    page = st.radio(
        "Navigate to:",
        ["🏠 Overview", "🌸 Q1 — DCGAN & WGAN-GP", "🖌️ Q2 — Pix2Pix", "🔄 Q3 — CycleGAN"],
        index=0,
    )

    st.divider()
    st.markdown("**Device:** " + ("🚀 CUDA/GPU" if DEVICE.type == "cuda" else "💻 CPU"))

    # Pre-load status
    dcgan_ok = os.path.exists(MODEL_PATHS["dcgan"])
    wgan_ok = os.path.exists(MODEL_PATHS["wgan"])
    p2p_ok = os.path.exists(MODEL_PATHS["pix2pix_g"])
    cg_ok = os.path.exists(MODEL_PATHS["cyclegan"])

    st.markdown("**Model Status:**")
    show_model_status("DCGAN", dcgan_ok)
    show_model_status("WGAN-GP", wgan_ok)
    show_model_status("Pix2Pix", p2p_ok)
    show_model_status("CycleGAN", cg_ok)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="gradient-text">🎨 GenAI Assignment 3 — GAN Showcase</p>', unsafe_allow_html=True)
    # st.markdown("##### *22F-3858 Faizan Tariq & 22F-3060 Hassaan Ch*")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌸 Question 1")
        st.markdown("**DCGAN & WGAN-GP**")
        st.markdown("Anime face generation trained on 21,551 faces for 50 epochs each. Compare standard DCGAN vs Wasserstein GAN with gradient penalty.")
        st.markdown('<span class="badge">64×64 output</span><span class="badge">z=100</span><span class="badge">50 epochs</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖌️ Question 2")
        st.markdown("**Pix2Pix (Sketch → Color)**")
        st.markdown("Anime sketch colorization using U-Net generator + PatchGAN discriminator. Trained on 5,000 sketch-color pairs for 30 epochs.")
        st.markdown('<span class="badge">256×256</span><span class="badge">U-Net</span><span class="badge">30 epochs</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Question 3")
        st.markdown("**CycleGAN (Sketch ↔ Photo)**")
        st.markdown("Unpaired domain translation between sketches and photos using cycle-consistency loss. Trained for 30 epochs with ResNet-6 generators on the Sketchy dataset.")
        st.markdown('<span class="badge">128×128</span><span class="badge">ResNet-6</span><span class="badge">30 epochs</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Training Results Summary")

    m_cols = st.columns(4)
    metrics = [
        ("DCGAN Loss G", "7.41", "Final epoch"),
        ("WGAN-GP Epochs", "50", "Full training"),
        ("Pix2Pix Best G", "~14.37", "Epoch 11"),
        ("CycleGAN SSIM", "0.9518", "Photo cycle"),
    ]
    for col, (label, val, sub) in zip(m_cols, metrics):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{label}<br><small>{sub}</small></div></div>', unsafe_allow_html=True)

    st.divider()
    st.info("👈 Select a question from the sidebar to interact with the models.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Q1: DCGAN & WGAN-GP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌸 Q1 — DCGAN & WGAN-GP":
    st.markdown('<p class="gradient-text">🌸 Question 1 — Anime Face Generation</p>', unsafe_allow_html=True)
    st.markdown("Generate anime face images using trained DCGAN and WGAN-GP generators.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎲 Generate Images", "📋 Architecture", "📈 Training Info"])

    with tab1:
        col_ctrl, col_out = st.columns([1, 2])

        with col_ctrl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Controls")
            model_choice = st.selectbox("Select Model", ["DCGAN", "WGAN-GP"])
            n_images = st.slider("Number of images", 1, 16, 8)
            seed = st.number_input("Random seed (0 = random)", min_value=0, max_value=9999, value=0)
            generate_btn = st.button("✨ Generate Anime Faces", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_out:
            if generate_btn:
                with st.spinner("Generating..."):
                    if model_choice == "DCGAN":
                        model = load_dcgan()
                        model_name = "DCGAN"
                    else:
                        model = load_wgan()
                        model_name = "WGAN-GP"

                    if model is None:
                        st.error(f"⚠️ {model_name} model file not found at `{MODEL_PATHS['dcgan' if model_choice == 'DCGAN' else 'wgan']}`")
                    else:
                        if seed > 0:
                            torch.manual_seed(seed)
                        z = torch.randn(n_images, 100, 1, 1, device=DEVICE)
                        with torch.no_grad():
                            out = model(z)

                        imgs = [tensor_to_pil(out[i]) for i in range(n_images)]

                        st.markdown(f"**{model_name} Generated Images** ({n_images} samples)")
                        cols_per_row = 4
                        for row_start in range(0, n_images, cols_per_row):
                            row_imgs = imgs[row_start: row_start + cols_per_row]
                            cols = st.columns(len(row_imgs))
                            for c, img in zip(cols, row_imgs):
                                c.image(img, width='stretch')

                        # Download strip
                        if n_images > 1:
                            strip_w = 64 * min(n_images, 8)
                            strip = Image.new("RGB", (strip_w, 64))
                            for i, img in enumerate(imgs[:8]):
                                strip.paste(img.resize((64, 64)), (i * 64, 0))
                            st.download_button(
                                "⬇️ Download first 8 as strip",
                                img_to_bytes(strip),
                                file_name=f"{model_name.lower()}_strip.png",
                                mime="image/png",
                            )

    with tab2:
        st.markdown('<p class="section-header">Generator Architecture</p>', unsafe_allow_html=True)
        st.code("""
class Generator(nn.Module):
    def __init__(self, z_input=100, feature_map=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_input, feature_map*8, 4, 1, 0),  # 4×4
            nn.BatchNorm2d(feature_map*8),  nn.ReLU(True),
            nn.ConvTranspose2d(feature_map*8, feature_map*4, 4, 2, 1),  # 8×8
            nn.BatchNorm2d(feature_map*4),  nn.ReLU(True),
            nn.ConvTranspose2d(feature_map*4, feature_map*2, 4, 2, 1),  # 16×16
            nn.BatchNorm2d(feature_map*2),  nn.ReLU(True),
            nn.ConvTranspose2d(feature_map*2, feature_map,   4, 2, 1),  # 32×32
            nn.BatchNorm2d(feature_map),    nn.ReLU(True),
            nn.ConvTranspose2d(feature_map, 3, 4, 2, 1),                # 64×64
            nn.Tanh()
        )
""", language="python")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**DCGAN** uses standard BCE loss with binary labels.")
        with c2:
            st.markdown("**WGAN-GP** replaces BCE with Wasserstein distance + gradient penalty.")

    with tab3:
        st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
        data = {
            "Dataset": "Anime Faces (Kaggle)",
            "Image Size": "64×64 RGB",
            "Batch Size": 32,
            "Epochs": 50,
            "Latent Dim (z)": 100,
            "Optimizer": "Adam (lr=0.0002, β=(0.5,0.999))",
            "DCGAN Loss": "BCEWithLogitsLoss",
            "WGAN-GP λ": 10,
        }
        for k, v in data.items():
            col_k, col_v = st.columns([1, 2])
            col_k.markdown(f"**{k}**")
            col_v.markdown(str(v))

        st.divider()
        st.markdown("**Final Training Losses (DCGAN):**")
        st.markdown("- Epoch 50 — `Loss_D: 0.0857` | `Loss_G: 7.4118`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Q2: Pix2Pix
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖌️ Q2 — Pix2Pix":
    st.markdown('<p class="gradient-text">🖌️ Question 2 — Sketch to Color (Pix2Pix)</p>', unsafe_allow_html=True)
    st.markdown("Upload an anime sketch and let the Pix2Pix model colorize it!")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎨 Colorize Sketch", "📋 Architecture", "📈 Training Info"])

    with tab1:
        col_in, col_out = st.columns(2)

        with col_in:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Input Sketch")
            st.markdown("Upload a grayscale or line-art anime sketch (PNG/JPG). For best results use **256×256** sketches similar to the training data.")
            uploaded = st.file_uploader("Choose sketch image", type=["png", "jpg", "jpeg"])

            if uploaded:
                input_img = Image.open(uploaded).convert("RGB")
                st.image(input_img, caption="Uploaded sketch", width='stretch')
                colorize_btn = st.button("✨ Colorize!", width='stretch')
            else:
                colorize_btn = False
                st.info("👆 Upload a sketch to get started.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_out:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Colorized Output")
            if uploaded and colorize_btn:
                with st.spinner("Colorizing..."):
                    model = load_pix2pix()
                    if model is None:
                        st.error(f"⚠️ Pix2Pix model not found at `{MODEL_PATHS['pix2pix_g']}`")
                    else:
                        inp_tensor = pil_to_tensor(input_img, 256)
                        with torch.no_grad():
                            out_tensor = model(inp_tensor)
                        out_img = tensor_to_pil(out_tensor[0])
                        st.image(out_img, caption="Pix2Pix colorized output", width='stretch')
                        st.download_button(
                            "⬇️ Download Result",
                            img_to_bytes(out_img),
                            file_name="pix2pix_colorized.png",
                            mime="image/png",
                        )
            else:
                st.markdown("*Output will appear here after colorization.*")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-header">U-Net Generator Architecture</p>', unsafe_allow_html=True)
        st.markdown("""
The generator follows a **U-Net** design with skip connections between encoder and decoder layers:

- **Encoder** (8 downsampling blocks): 256→128→64→32→16→8→4→2→1
- **Bottleneck**: 512 channels, 1×1 spatial
- **Decoder** (8 upsampling blocks with skip connections): 1→2→...→256
- **Output**: 3-channel RGB image with `Tanh` activation
        """)

        st.code("""
# Encoder
e1 = Conv2d(3,   64,  4,2,1) + LeakyReLU   # 256→128
e2 = Conv2d(64,  128, 4,2,1) + BN + LReLU  # 128→64
...
e8 = Conv2d(512, 512, 4,2,1)               # 2→1 (bottleneck)

# Decoder with skip connections
d1 = ConvTranspose + BN + ReLU + Dropout
d2 = ConvTranspose(cat[d1,e7]) ...
...
d8 = ConvTranspose(cat[d7,e1]) + Tanh       # Output 256
""", language="python")

        st.markdown('<p class="section-header">PatchGAN Discriminator</p>', unsafe_allow_html=True)
        st.markdown("Takes the concatenation of sketch + real/fake image as input and produces a **30×30 patch** discriminator map.")

    with tab3:
        st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
        data = {
            "Dataset": "Anime Sketch–Color pairs (Kaggle)",
            "Image Size": "256×256 RGB",
            "Batch Size": 16,
            "Epochs": 30,
            "LR": "0.0002, β=(0.5, 0.999)",
            "λ L1": 100,
            "Generator Loss": "BCE + λ·L1",
        }
        for k, v in data.items():
            col_k, col_v = st.columns([1, 2])
            col_k.markdown(f"**{k}**")
            col_v.markdown(str(v))

        st.divider()
        st.markdown("**Training Loss progression:**")
        epochs = [1, 5, 10, 15, 20, 25, 30]
        g_losses = [29.90, 18.98, 15.18, 12.0, 10.5, 9.2, 8.4]
        d_losses = [0.42, 0.53, 0.52, 0.51, 0.50, 0.49, 0.49]
        import pandas as pd
        df = pd.DataFrame({"Epoch": epochs, "Generator Loss": g_losses, "Discriminator Loss": d_losses})
        st.line_chart(df.set_index("Epoch"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Q3: CycleGAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Q3 — CycleGAN":
    st.markdown('<p class="gradient-text">🔄 Question 3 — CycleGAN (Sketch ↔ Photo)</p>', unsafe_allow_html=True)
    st.markdown("Unpaired image-to-image translation between sketch and photo domains.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔄 Translate Image", "📋 Architecture", "📈 Training Info"])

    with tab1:
        direction = st.radio(
            "Translation direction:",
            ["Sketch → Photo  (G_AB)", "Photo → Sketch  (G_BA)"],
            horizontal=True,
        )

        col_in, col_out = st.columns(2)

        domain_label = "sketch" if "Sketch → Photo" in direction else "photo"
        output_label = "photo" if "Sketch → Photo" in direction else "sketch"

        with col_in:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Input {domain_label.title()}")
            st.markdown(f"Upload a {domain_label} image (PNG/JPG). Images will be resized to **128×128**.")
            uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
            if uploaded:
                input_img = Image.open(uploaded).convert("RGB")
                st.image(input_img, caption=f"Input {domain_label}", width='stretch')
                translate_btn = st.button(f"✨ Translate to {output_label}!", width='stretch')
            else:
                translate_btn = False
                st.info(f"👆 Upload a {domain_label} image.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_out:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Output {output_label.title()}")
            if uploaded and translate_btn:
                with st.spinner("Translating..."):
                    g_ab, g_ba = load_cyclegan()
                    if g_ab is None:
                        st.error(f"⚠️ CycleGAN models not found. Check `question_3_model/` directory.")
                    else:
                        model = g_ab if "Sketch → Photo" in direction else g_ba
                        inp_tensor = pil_to_tensor(input_img, 128)
                        with torch.no_grad():
                            out_tensor = model(inp_tensor)
                        out_img = tensor_to_pil(out_tensor[0])
                        st.image(out_img, caption=f"CycleGAN {output_label} output", width='stretch')
                        st.download_button(
                            "⬇️ Download Result",
                            img_to_bytes(out_img),
                            file_name=f"cyclegan_{output_label}.png",
                            mime="image/png",
                        )
            else:
                st.markdown("*Output will appear here after translation.*")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-header">ResNet Generator Architecture</p>', unsafe_allow_html=True)
        st.markdown("""
Each CycleGAN generator uses a **ResNet-6** architecture:

1. **Encoder** — 3 Conv layers (reflection padding, InstanceNorm, ReLU)
2. **Residual Blocks** — 6 ResBlocks for feature transformation
3. **Decoder** — 2 ConvTranspose layers (upsample back to input size)
4. **Output** — `Tanh` activation → pixel values in [-1, 1]
        """)

        st.code("""
class Generator(nn.Module):
    # Encoder: 3 → 64 → 128 → 256 channels (downsample x2)
    # 6× ResBlock(256)
    # Decoder: 256 → 128 → 64 → 3 channels (upsample x2)
    # Final: ReflectionPad + Conv7 + Tanh
""", language="python")

        st.markdown('<p class="section-header">PatchGAN Discriminators</p>', unsafe_allow_html=True)
        st.markdown("Two discriminators **D_A** and **D_B** — one per domain — each using 4-layer PatchGAN with InstanceNorm.")

        st.markdown('<p class="section-header">Loss Functions</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Adversarial (MSE)**\n\nReplaces BCE with least-squares for more stable training.")
        with c2:
            st.markdown("**Cycle Consistency (L1)**\n\nλ = 10 ensures G_BA(G_AB(x)) ≈ x")
        with c3:
            st.markdown("**Identity (L1)**\n\nλ_id = 5 preserves colour when input is already in target domain")

    with tab3:
        st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
        data = {
            "Dataset": "Sketchy Dataset (unpaired sketch + photo)",
            "Image Size": "128×128 RGB",
            "Batch Size": 4,
            "Epochs": 30,
            "LR": "0.0002, β=(0.5, 0.999)",
            "λ Cycle": 10,
            "λ Identity": 5,
            "ResBlocks": 6,
            "Adversarial Loss": "MSE (LSGAN)",
            "Replay Buffer": "50 images",
        }
        for k, v in data.items():
            col_k, col_v = st.columns([1, 2])
            col_k.markdown(f"**{k}**")
            col_v.markdown(str(v))

        st.divider()
        st.markdown("**Final Evaluation Metrics (Epoch 30):**")
        m_cols = st.columns(2)
        with m_cols[0]:
            st.markdown('<div class="metric-box"><div class="metric-value">0.8244</div><div class="metric-label">Sketch Cycle SSIM</div></div>', unsafe_allow_html=True)
        with m_cols[1]:
            st.markdown('<div class="metric-box"><div class="metric-value">0.9518</div><div class="metric-label">Photo Cycle SSIM</div></div>', unsafe_allow_html=True)
