import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os
import sys
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
import io

# Add project root to sys.path so 'app' can be resolved when running via streamlit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.face import FaceProcessor

st.set_page_config(page_title="FaceTrace Dashboard", layout="wide")

st.title("FaceTrace Web Dashboard")
st.markdown("Upload a crowd photo, select your target, and trace their digital footprint.")
st.divider()

@st.cache_resource
def get_face_processor():
    # Cache the face processor so it doesn't reload the 300MB model on every UI interaction
    return FaceProcessor()

fp = get_face_processor()

# Create a 20% / 80% split for the entire dashboard
left_col, right_col = st.columns([1, 4])

with left_col:
    st.subheader("⚙️ Search Config")
    engine = st.selectbox("Search Engine", options=["all", "lens", "yandex", "serpapi"])
    threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01)
    target_url = st.text_input("Target URL (Optional)", help="Verify against a specific post/video")
    handle = st.text_input("Username Sweep (Optional)", help="Sweep a specific handle across platforms")
    skip_blockchain = st.checkbox("Skip Blockchain Notarization", value=True)

with right_col:
    st.subheader("📷 Image Analysis")
    
    # File Uploader in the main container
    uploaded_file = st.file_uploader("Drag and drop your crowd photo here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file to temp dir
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Process image to find all faces
        with st.spinner("Analyzing faces..."):
            annotated_img, face_data = fp.annotate_and_extract_all_faces(tmp_path)
        
        if len(face_data) == 0:
            st.error("No faces detected in the image.")
        else:
            # Create the Tabs layout
            tab_visuals, tab_terminal = st.tabs(["📷 Visuals", "🖥️ Terminal"])
            
            with tab_visuals:
                # Render the annotated image inside a container for perfect scaling
                with st.container(border=True):
                    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_img, caption=f"Found {len(face_data)} faces", use_container_width=True)
                
                # Placeholder for the clean Option A status
                status_placeholder = st.empty()
            
            # Now go BACK to the left column to append the target selection and button
            with left_col:
                st.divider()
                st.subheader("🎯 Target Selection")
                target_idx = st.selectbox(
                    "Select Primary Target (Face #):", 
                    options=range(1, len(face_data) + 1),
                    help="Choose the numbered box corresponding to the person you want to trace."
                )

                run_btn = st.button("🚀 Run FaceTrace", type="primary", use_container_width=True)

            if run_btn:
                # Crop the selected face to bypass the multi-face CLI limitation
                selected_face = face_data[target_idx - 1]
                x1, y1, x2, y2 = selected_face['bbox']
                
                # Add a 60% margin around the face to ensure the detector has enough context (neck, hair)
                margin = 0.60
                pad_w = int((x2 - x1) * margin)
                pad_h = int((y2 - y1) * margin)
                
                orig_img = cv2.imread(tmp_path)
                h, w = orig_img.shape[:2]
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w, x2 + pad_w)
                cy2 = min(h, y2 + pad_h)
                
                cropped_face = orig_img[cy1:cy2, cx1:cx2]
                
                # Save the cropped face to a new temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as cropped_tmp:
                    cv2.imwrite(cropped_tmp.name, cropped_face)
                    cropped_path = cropped_tmp.name

                import subprocess
                import re
                
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

                with tab_terminal:
                    st.caption("Raw execution logs are streaming below...")
                    log_container = st.empty()

                with status_placeholder.status("Initializing Pipeline...", expanded=True) as status:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    
                    full_log = ""
                    # Regex to match step headers like "[1/7] FACE DETECTION"
                    step_regex = re.compile(r"\[(\d+/\d+)\]\s+(.*)")
                    
                    for line in iter(process.stdout.readline, ''):
                        full_log += line
                        
                        # Strip ANSI colors for parsing
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
                        
                        # Check for steps
                        step_match = step_regex.search(clean_line)
                        if step_match:
                            status.update(label=f"Step {step_match.group(1)}: {step_match.group(2)}")
                            st.write(f"⏳ {step_match.group(2)}...")
                        elif "✓" in clean_line:
                            # Highlight successful sub-steps in the dropdown
                            st.write(f"✅ {clean_line.replace('✓', '').strip()}")
                        elif "✗" in clean_line:
                            st.write(f"❌ {clean_line.replace('✗', '').strip()}")
                        
                        log_container.code(full_log, language="bash")
                    
                    process.stdout.close()
                    process.wait()
                    
                    if process.returncode == 0:
                        status.update(label="FaceTrace Complete!", state="complete", expanded=False)
                    else:
                        status.update(label="Pipeline Failed", state="error", expanded=True)
                
                with tab_visuals:
                    if process.returncode == 0:
                        st.success("Target successfully traced! Check Terminal tab for complete log details.")
                    else:
                        st.error(f"Pipeline exited with error code {process.returncode}. See Terminal tab.")
    else:
        st.info("Upload an image to begin.")
