import os
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

class AIEngine:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        # Load the custom coach instructions
        try:
            with open(os.path.join(os.path.dirname(__file__), 'coach_instructions.txt'), 'r', encoding='utf-8') as f:
                self.chat_prompt = f.read()
        except FileNotFoundError:
            self.chat_prompt = "You are FitAI, a concise and professional fitness coach."
            
        self.system_prompt = "You are FitAI, a strict biomechanics coach. Receive JSON data about exercise. Keep responses under 50 words. Focus on form correction."

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
