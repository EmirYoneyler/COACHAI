import streamlit as st
import cv2
import numpy as np
import sys
try:
    from src.motion_tracker import MotionTracker
except ImportError:
    MotionTracker = None
except Exception as e:
    st.error(f"Error importing MotionTracker: {e}")
    MotionTracker = None

try:
    from src.ai_engine import AIEngine
except ImportError:
    st.error("Error importing AIEngine. Please check dependencies.")
    AIEngine = None
except Exception as e:
    st.error(f"Error importing AIEngine: {e}")
    AIEngine = None

# Page Config
st.set_page_config(page_title="FitAI Coach", layout="wide")

# Debug Info
st.sidebar.write(f"Python Version: {sys.version.split()[0]}")

# Initialize Classes
if 'tracker' not in st.session_state:
    if MotionTracker:
        try:
            st.session_state.tracker = MotionTracker()
        except Exception as e:
            st.error(f"Failed to initialize MotionTracker: {e}")
            st.session_state.tracker = None
    else:
        st.session_state.tracker = None

if 'ai_engine' not in st.session_state:
    if AIEngine:
        try:
            st.session_state.ai_engine = AIEngine()
        except Exception as e:
            st.error(f"Failed to initialize AIEngine: {e}")
            st.session_state.ai_engine = None
    else:
        st.session_state.ai_engine = None

st.title("🏋️ FitAI: Your AI Fitness Coach")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose Mode", ["Motion Analysis", "AI Planner", "Chat"])

if page == "Motion Analysis":
    st.header("Real-Time Squat Analysis")
    
    if st.session_state.tracker is None:
        st.error("⚠️ Motion Tracker failed to load.")
        st.write("Debug details:")
        try:
            import mediapipe as mp
            st.success("MediaPipe is importable.")
        except ImportError as e:
            st.error(f"MediaPipe Import Error: {e}")
            
        st.info("If you see this, please try running 'start.bat' again to install dependencies.")
    else:
        st.write("Enable your camera to start tracking.")
        
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        while run:
            _, frame = camera.read()
            if frame is None:
                st.warning("Camera not detected.")
                break
                
            # Process Frame
            frame, data = st.session_state.tracker.process_frame(frame)
            counter = data['reps']
            state = data['state']
            
            # Display Output
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
            # Metrics
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Reps", counter)
            kpi2.metric("State", state)
        
        camera.release()

elif page == "AI Planner":
    st.header("Personalized Workout & Diet Plan")
    
    if st.session_state.ai_engine is None:
        st.error("AI Engine is not initialized. Please check your API key and dependencies.")
    else:
        with st.form("stats_form"):
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=175)
            goal = st.selectbox("Goal", ["Lose Weight", "Build Muscle", "Maintain"])
            activity = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])
            
            submitted = st.form_submit_button("Generate Plan")
            
            if submitted:
                stats = {
                    "weight": weight,
                    "height": height,
                    "goal": goal,
                    "activity_level": activity
                }
                with st.spinner("Generating your plan..."):
                    plan = st.session_state.ai_engine.generate_plan(stats)
                    st.markdown(plan)

elif page == "Chat":
    st.header("Ask FitAI")
    
    if st.session_state.ai_engine is None:
        st.error("AI Engine is not initialized. Please check your API key and dependencies.")
    else:
        user_input = st.text_input("Ask about fitness, nutrition, or health:")
        
        if st.button("Send"):
            if user_input:
                with st.spinner("Thinking..."):
                    response = st.session_state.ai_engine.get_chat_response(user_input)
                    st.write(response)
