import os
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

class AIEngine:
    def __init__(self):
        # API Key (Loaded from environment)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        # Load the custom coach instructions
        try:
            with open(os.path.join(os.path.dirname(__file__), 'coach_instructions.txt'), 'r', encoding='utf-8') as f:
                self.chat_prompt = f.read()
        except FileNotFoundError:
            self.chat_prompt = "You are FitAI, a concise and professional fitness coach."
            
        self.system_prompt = "You are FitAI, a strict biomechanics coach. Receive JSON data about exercise. Provide detailed form correction feedback."

    def get_exercise_parameters(self, exercise_name: str) -> dict:
        """
        Asks the AI for the biomechanical parameters to track a new exercise.
        Returns a JSON dict with:
        - key_landmarks: [list of 3 pose landmarks to form an angle, e.g. ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST']]
        - thresholds: {'down': angle, 'up': angle} - indicating the range of motion
        - description: "Short description"
        """
        if not self.client:
            return None

        prompt = f"""
        Provide the biomechanical tracking parameters for the exercise: '{exercise_name}'.
        Return ONLY valid JSON. No markdown.
        Format:
        {{
            "landmarks": ["point_A", "point_B", "point_C"],
            "thresholds": {{"min": number, "max": number}},
            "mode": "min_max" or "max_min", 
            "description": "Short description of the movement."
        }}
        - landmarks must be 3 specific MediaPipe Pose Landmarks keys (e.g. LEFT_HIP, LEFT_KNEE, LEFT_ANKLE) that best define the repetition.
        - thresholds defines the angle values at the extremes of the movement.
        - mode 'min_max' means start low, go high to count (like lateral raise), 'max_min' means start high, go low (like squat).
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a computer vision expert. You output strictly JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # Clean possible markdown code blocks
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            import json
            return json.loads(content)
        except Exception as e:
            print(f"Error getting parameters: {e}")
            return None

    def analyze_form(self, motion_data: dict) -> str:
        """
        Analyzes motion data using GPT-4o-mini.
        """
        if not self.client:
            return "Error: OpenAI API Key not found."

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(motion_data)}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        except RateLimitError:
            return "Error: OpenAI API Quota Exceeded. Please check your billing details at platform.openai.com."
        except Exception as e:
            return f"Error analyzing form: {str(e)}"

    def analyze_recorded_set(self, data: dict) -> str:
        """
        Analyzes a full set of recorded motion data.
        """
        if not self.client:
            return "Error: OpenAI API Key not found."

        system_msg = "You are a strict Strength Coach. Output ONLY the Form Score, 3 specific cues, and a weight recommendation."
        
        prompt = f"""
        Analyze this set of {data.get('exercise_name', 'Exercise')}.
        Data: {str(data.get('frames', []))}
        Keys: i=frame_index, a=angle, s=stage, l=landmarks(x,y).
        
        REQUIRED OUTPUT FORMAT (No other text):
        Form Score: [0-10]/10
        Cues for Improvement:
        - [Cue 1]
        - [Cue 2]
        - [Cue 3]
        
        Recommendation:
        - If Score < 7: "Your form is breaking down. Lower the weight immediately to prevent injury."
        - If Score >= 7: "Good weight management. Focus on controlling the eccentric phase."
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except RateLimitError:
            return "Error: Quota Exceeded."
        except Exception as e:
            # Fallback for Context Length Error - try to truncate
            if "context_length_exceeded" in str(e):
                return "Error: Recording too long. Please try a shorter set (max 10-15 reps)."
            return f"Error analyzing set: {str(e)}"

    def generate_plan(self, user_stats: dict) -> str:
        """
        Generates a workout/nutrition plan based on user stats.
        """
        if not self.client:
            return "Error: OpenAI API Key not found. Please check your .env file."

        prompt = f"""
        You are FitAI, an expert fitness coach. Create a concise plan for a user with these stats:
        - Weight: {user_stats.get('weight')} kg
        - Height: {user_stats.get('height')} cm
        - Goal: {user_stats.get('goal')}
        - Activity Level: {user_stats.get('activity_level')}
        
        Provide:
        1. Daily Calorie Target
        2. Macro Split (Protein/Carbs/Fats)
        3. 3 Key Exercises recommended
        
        Keep it under 150 words. Use bullet points.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful fitness assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except RateLimitError:
            return "Error: OpenAI API Quota Exceeded. Please check your billing details at platform.openai.com."
        except Exception as e:
            return f"Error generating plan: {str(e)}"

    def get_chat_response(self, user_message: str, chat_history: list = None) -> str:
        """
        Handles general chat queries with context.
        """
        if not self.client:
             return "Error: OpenAI API Key not found."

        # Use the loaded chat prompt as the system message
        messages = [{"role": "system", "content": self.chat_prompt}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=150
            )
            return response.choices[0].message.content
        except RateLimitError:
            return "Error: OpenAI API Quota Exceeded. Please check your billing details at platform.openai.com."
        except Exception as e:
            return f"Error: {str(e)}"
