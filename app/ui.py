import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os
import sys
import tempfile
import subprocess
import re

# Add project root to sys.path so 'app' can be resolved
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.face import FaceProcessor

# ==========================================
# Page Config & Cyber-Sleek CSS
# ==========================================
st.set_page_config(page_title="Social Detective Dashboard", layout="wide", page_icon="🕵️‍♂️")

st.markdown("""
<style>
/* Deep Dark Mode Background */
.stApp {
    background-color: #090a0f;
    color: #e0e6ed;
}

/* Glassmorphism Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 20, 25, 0.85) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0, 238, 255, 0.2);
}

/* Neon Primary Buttons */
[data-testid="baseButton-primary"] {
    background: transparent !important;
    border: 1px solid #00eeff !important;
    color: #00eeff !important;
    box-shadow: 0 0 10px rgba(0, 238, 255, 0.2);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="baseButton-primary"]:hover {
    background: rgba(0, 238, 255, 0.15) !important;
    box-shadow: 0 0 20px rgba(0, 238, 255, 0.6);
    color: #ffffff !important;
}

/* Headers */
h1, h2, h3 {
    color: #00eeff !important;
    text-shadow: 0 0 15px rgba(0, 238, 255, 0.4);
    font-family: 'Courier New', Courier, monospace;
}

/* Glowing text inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(0,0,0,0.5) !important;
    border: 1px solid rgba(0, 238, 255, 0.3) !important;
    color: #e0e6ed !important;
}
.stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: #00eeff !important;
    box-shadow: 0 0 10px rgba(0, 238, 255, 0.5) !important;
}

/* Terminal logs styling (st.code) */
code {
    background: rgba(0, 5, 10, 0.8) !important;
    color: #00eeff !important;
    border: 1px solid rgba(0, 238, 255, 0.2);
    box-shadow: inset 0 0 10px rgba(0, 238, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)


st.title("Social Detective OSINT Dashboard")
st.markdown("Upload a photo and trace their digital footprint across the web.")
st.divider()

# ==========================================
# Sidebar Configuration
# ==========================================
with st.sidebar:
    st.header("⚙️ Search Config")
    engine = st.selectbox("Search Engine", options=["all", "lens", "yandex", "serpapi"])
    threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01)
    target_url = st.text_input("Target URL (Optional)", help="Verify against a specific post/video")
    handle = st.text_input("Username Sweep (Optional)", help="Sweep a specific handle across platforms")
    skip_blockchain = st.checkbox("Skip Blockchain Notarization", value=True)
    context_text = st.text_area("Context Clues (Optional)", help="Add any known context to guide the search (e.g., 'Stanford University 2026')")

@st.cache_resource
def get_face_processor():
    return FaceProcessor()

fp = get_face_processor()

# ==========================================
# Main Workflow: Single Target Hunt
# ==========================================
st.subheader("📷 Target Initialization")
uploaded_main = st.file_uploader("Upload a photo of the individual", type=["jpg", "jpeg", "png"], key="single_up")

if uploaded_main is not None:
    # Save main file to temp dir
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_main.getvalue())
        tmp_path = tmp.name

    with st.spinner("Extracting facial biometrics..."):
        try:
            faces = fp.detect_faces(tmp_path)
            # Manually draw bounding boxes and build face_data
            img_bgr = cv2.imread(tmp_path)
            face_data = []
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                face_data.append({"bbox": (x1, y1, x2, y2), "embedding": face.embedding})
                
                # Add 15% visual padding so the box doesn't overlap the face edges
                img_h, img_w = img_bgr.shape[:2]
                v_pad_w = int((x2 - x1) * 0.15)
                v_pad_h = int((y2 - y1) * 0.15)
                vx1 = max(0, x1 - v_pad_w)
                vy1 = max(0, y1 - v_pad_h)
                vx2 = min(img_w, x2 + v_pad_w)
                vy2 = min(img_h, y2 + v_pad_h)
                
                # Draw neon cyan box (BGR: 255, 238, 0)
                cv2.rectangle(img_bgr, (vx1, vy1), (vx2, vy2), (255, 238, 0), 2)
                label = f"#{i + 1}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (vx1, vy1 - 20), (vx1 + w, vy1), (255, 238, 0), -1)
                cv2.putText(img_bgr, label, (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            annotated_img = img_bgr
        except Exception as e:
            st.error(f"Error processing faces: {e}")
            face_data = []
    
    if len(face_data) == 0:
        st.error("No faces detected in the image.")
    else:
        st.subheader("🎯 Target Selection")
        target_idx = None
        
        col_img, col_sel = st.columns([2, 1])
        
        with col_img:
            # Render the annotated image
            with st.container(border=True):
                rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption=f"Found {len(face_data)} faces")
        
        with col_sel:
            if len(face_data) == 1:
                st.info("Single face detected. Automatically selected.")
                target_idx = 1
            else:
                target_idx = st.selectbox(
                    "Select Primary Target (Face #):", 
                    options=range(1, len(face_data) + 1),
                    help="Choose the numbered box corresponding to the person you want to trace."
                )

            st.divider()
            run_btn = st.button("🚀 Execute Social Detective", type="primary", use_container_width=True)

        # ==========================================
        # Execution & Terminal Tabs
        # ==========================================
        if run_btn:
            # Crop the selected face
            selected_face = face_data[target_idx - 1]
            x1, y1, x2, y2 = selected_face['bbox']
            
            margin = 0.70
            pad_w = int((x2 - x1) * margin)
            pad_h = int((y2 - y1) * margin)
            
            orig_img = cv2.imread(tmp_path)
            h, w = orig_img.shape[:2]
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)
            
            cropped_face = orig_img[cy1:cy2, cx1:cx2]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as cropped_tmp:
                cv2.imwrite(cropped_tmp.name, cropped_face)
                cropped_path = cropped_tmp.name

            # Build CLI command
            cmd = [
                sys.executable, "-m", "app.main",
                "--image", cropped_path,
                "--engine", engine,
                "--threshold", str(threshold)
            ]
            if target_url:
                cmd.extend(["--target", target_url])
            if handle:
                cmd.extend(["--handle", handle])
            if skip_blockchain:
                cmd.append("--skip-blockchain")
            if context_text.strip():
                cmd.extend(["--context", context_text.strip()])

            st.divider()
            st.subheader("🔍 Analysis Dashboard")
            tab_visuals, tab_terminal = st.tabs(["🚀 Execution Pipeline", "⚙️ Under the Hood (Logs)"])

            with tab_terminal:
                st.caption("Raw execution logs streaming securely...")
                log_container = st.empty()
            
            # CRITICAL FIX: The Streamlit UI holds the FaceProcessor ONNX model in GPU memory.
            # If we spawn the backend pipeline while the UI holds the GPU, the backend crashes with 
            # a STATUS_STACK_BUFFER_OVERRUN (0xC0000409) due to VRAM collision.
            # We MUST clear the UI's FaceProcessor from memory before launching the backend!
            get_face_processor.clear()
            import gc
            gc.collect()

            with tab_visuals:
                status_placeholder = st.empty()
                with status_placeholder.status(f"Hunting for Target #{target_idx}...", expanded=True) as status:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    
                    full_log = ""
                    step_regex = re.compile(r"\[(\d+/\d+)\]\s+(.*)")
                    
                    for line in iter(process.stdout.readline, ''):
                        full_log += line
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
                        
                        step_match = step_regex.search(clean_line)
                        if step_match:
                            status.update(label=f"Step {step_match.group(1)}: {step_match.group(2)}")
                            st.write(f"⏳ {step_match.group(2)}...")
                        elif "✓" in clean_line:
                            st.write(f"✅ {clean_line.replace('✓', '').strip()}")
                        elif "✗" in clean_line:
                            st.write(f"❌ {clean_line.replace('✗', '').strip()}")
                        
                        log_container.code(full_log, language="bash")
                    
                    process.stdout.close()
                    process.wait()
                    
                    if process.returncode == 0:
                        status.update(label="Social Detective Complete!", state="complete", expanded=False)
                        st.success("Target successfully traced! Check 'Under the Hood' tab for log details.")
                    else:
                        status.update(label="Pipeline Failed", state="error", expanded=True)
                        st.error(f"Pipeline exited with error code {process.returncode}. See 'Under the Hood' tab.")
