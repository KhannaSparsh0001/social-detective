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
from app.context import ContextManager
from app.search import Candidate

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
            tab_visuals, tab_terminal = st.tabs(["📷 Analysis Dashboard", "⚙️ Under the Hood (Logs)"])
            
            with tab_visuals:
                # Render the annotated image inside a container for perfect scaling
                with st.container(border=True):
                    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_img, caption=f"Found {len(face_data)} faces")
                
                # Placeholder for the clean Option A status
                status_placeholder = st.empty()
            
            # Now go BACK to the left column to append the target selection and button
            with left_col:
                st.divider()
                st.subheader("🎯 Target Selection")
                
                # State management for multi-phase workflow
                if "phase_1_result" not in st.session_state:
                    st.session_state.phase_1_result = None
                if "context_tags" not in st.session_state:
                    st.session_state.context_tags = []
                
                target_idx = None
                if len(face_data) == 1:
                    st.info("Single face detected. Automatically selected.")
                    target_idx = 1
                else:
                    selection_mode = st.radio("Selection Mode", ["Manual Select", "Auto-Match Reference"])
                    if selection_mode == "Manual Select":
                        target_idx = st.selectbox(
                            "Select Primary Target (Face #):", 
                            options=range(1, len(face_data) + 1),
                            help="Choose the numbered box corresponding to the person you want to trace."
                        )
                    else:
                        ref_file = st.file_uploader("Upload Reference Photo (Single Person)", type=["jpg", "jpeg", "png"])
                        if ref_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as rtmp:
                                rtmp.write(ref_file.getvalue())
                            
                            with st.spinner("Auto-matching target..."):
                                try:
                                    ref_emb = fp.get_embedding(rtmp.name)
                                    best_idx = 0
                                    best_sim = -1.0
                                    for i, f_data in enumerate(face_data):
                                        emb = f_data['embedding']
                                        sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb))
                                        if sim > best_sim:
                                            best_sim = sim
                                            best_idx = i
                                    
                                    target_idx = best_idx + 1
                                    st.success(f"Auto-Matched to Face #{target_idx} ({best_sim*100:.1f}% similarity)")
                                except Exception:
                                    st.error("No valid face found in reference photo.")
                
                run_btn = False
                if target_idx is not None:
                    run_btn = st.button("🚀 Run FaceTrace", type="primary")

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

                with status_placeholder.status(f"Hunting for Target #{target_idx}...", expanded=True) as status:
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
                        st.session_state.phase_1_result = "success"
                    else:
                        status.update(label="Pipeline Failed", state="error", expanded=True)
                        st.session_state.phase_1_result = "failed"
                
                with tab_visuals:
                    if st.session_state.phase_1_result == "success":
                        st.success("Target successfully traced! Check 'Under the Hood' tab for log details.")
                    else:
                        st.error(f"Pipeline exited with error code {process.returncode}. See 'Under the Hood' tab.")
                        
            # Phase 2: Crowd Pivot UI
            if st.session_state.phase_1_result == "failed" and len(face_data) > 1:
                with left_col:
                    st.divider()
                    st.warning("⚠️ Target not found directly. Initiating Phase 2: Crowd Pivot!")
                    st.subheader("👥 Crowd Pivot Configuration")
                    
                    pivot_mode = st.radio(
                        "Select Background Faces to Scan:", 
                        ["Auto-Select Top 3", "Manual Select", "Search ALL Faces (High Rate Limit Risk)"]
                    )
                    
                    pivot_idxs = []
                    if pivot_mode == "Auto-Select Top 3":
                        other_faces = [(i, f) for i, f in enumerate(face_data) if i != target_idx - 1]
                        other_faces.sort(key=lambda x: (x[1]['bbox'][2]-x[1]['bbox'][0])*(x[1]['bbox'][3]-x[1]['bbox'][1]), reverse=True)
                        pivot_idxs = [i for i, f in other_faces[:3]]
                        st.info(f"Auto-selected {len(pivot_idxs)} largest background faces.")
                    elif pivot_mode == "Manual Select":
                        options = [f"Face #{i+1}" for i in range(len(face_data)) if i != target_idx - 1]
                        selected = st.multiselect("Select faces to pivot on:", options)
                        pivot_idxs = [int(s.split('#')[1])-1 for s in selected]
                    else:
                        st.error("WARNING: Searching all faces may trigger API rate bans.")
                        pivot_idxs = [i for i in range(len(face_data)) if i != target_idx - 1]

                    if st.button("🔍 Extract Crowd Context"):
                        with st.spinner("Extracting metadata from crowd profiles..."):
                            # Mocking candidate extraction for UI demonstration
                            # In full prod, we would run OSINT pipelines for each pivot_idx
                            dummy_candidates = [
                                Candidate(image_url="", source_url="", title="Attended #HackHazards 2026 at Stanford University", domain="stanford.edu"),
                                Candidate(image_url="", source_url="", title="Software Engineer @Google - @JohnDoe", domain="linkedin.com"),
                                Candidate(image_url="", source_url="", title="Team photo from the #AI retreat", domain="instagram.com")
                            ]
                            
                            cm = ContextManager()
                            st.session_state.context_tags = cm.process_and_suggest(dummy_candidates)
                            st.session_state.show_keyword_ui = True
                            
                # Keyword Selection Form
                if st.session_state.get('show_keyword_ui', False):
                    with tab_visuals:
                        st.subheader("🏷️ Context Keywords Extracted")
                        st.write("We extracted the following context tags from the background people. Select the most relevant tags to inject into the primary search:")
                        
                        with st.form("keyword_form"):
                            selected_tags = []
                            for tag in st.session_state.context_tags:
                                if st.checkbox(tag, value=True):
                                    selected_tags.append(tag)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                manual_submit = st.form_submit_button("🚀 Run Targeted Search")
                            with col2:
                                auto_submit = st.form_submit_button("🤖 Auto-Select Top 5 & Run")
                                
                            if manual_submit or auto_submit:
                                final_tags = selected_tags
                                if auto_submit:
                                    final_tags = st.session_state.context_tags[:5]
                                
                                context_str = ",".join(final_tags)
                                st.success(f"Running Phase 2 Search with context: `{context_str}`")
                                # Here we would trigger subprocess.Popen with --context
                                # e.g. cmd.extend(["--context", context_str])
                                
    else:
        st.info("Upload an image to begin.")
