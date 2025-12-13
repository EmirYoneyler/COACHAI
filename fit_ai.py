import json

class FitAI:
    def __init__(self):
        self.role = "Expert AI Fitness Coach and Kinesiologist"

    def process_input(self, input_data):
        """
        Distinguishes between Motion Data (JSON) and User Chat (String).
        """
        if isinstance(input_data, dict) and "exercise" in input_data:
            return self.analyze_motion(input_data)
        else:
            return self.handle_chat(input_data)

    def analyze_motion(self, data):
        """
        Analyzes motion data based on errors and provides real-time feedback.
        """
        exercise = data.get("exercise", "Unknown Exercise")
        errors = data.get("errors", [])
        
        # Acknowledge exercise
        response_parts = [f"Exercise: {exercise}."]

        # Calculate score (Simple heuristic: 10 - 2 points per error)
        score = max(0, 10 - (len(errors) * 2))

        if errors:
            # Explain fixes
            fixes = []
            for error in errors:
                error_lower = error.lower()
                if "knees caving" in error_lower:
                    fixes.append("Push knees outward.")
                elif "depth" in error_lower:
                    fixes.append("Squat deeper.")
                elif "back" in error_lower and "round" in error_lower:
                    fixes.append("Chest up, back straight.")
                else:
                    fixes.append(f"Fix: {error}.")
            
            response_parts.append(" ".join(fixes))
        else:
            response_parts.append("Perfect form.")

        response_parts.append(f"Score: {score}/10")
        
        final_response = " ".join(response_parts)
        
        # Ensure under 50 words
        if len(final_response.split()) > 50:
             final_response = " ".join(final_response.split()[:50]) + "..."
             
        return final_response

    def handle_chat(self, user_input):
        """
        Handles diet and planning questions.
        """
        # Simple keyword detection for demonstration
        user_input_lower = str(user_input).lower()
        
        if "plan" in user_input_lower or "stats" in user_input_lower:
            return (
                "Please provide your stats:\n"
                "- Weight\n"
                "- Height\n"
                "- Goal (Lose/Gain/Maintain)\n"
                "- Any blood test results?"
            )
        
        if "deficiency" in user_input_lower or "vitamin" in user_input_lower:
            if "vitamin d" in user_input_lower:
                return (
                    "**Vitamin D Deficiency Detected:**\n"
                    "- Fatty fish (Salmon, Tuna)\n"
                    "- Egg yolks\n"
                    "- Fortified foods (Milk, Cereal)\n"
                    "- Consider a supplement if advised by a doctor."
                )
        
        return "I am FitAI. How can I help with your training or diet today?"

# Example Usage
if __name__ == "__main__":
    bot = FitAI()

    # 1. Motion Data Example
    motion_data = {
        "exercise": "Squat",
        "rep_count": 8,
        "errors": ["Knees caving in", "Depth insufficient"],
        "avg_tempo": "2.5s"
    }
    print("--- Motion Data Output ---")
    print(bot.process_input(motion_data))
    print("\n")

    # 2. User Chat Example
    chat_input = "I have a Vitamin D deficiency."
    print("--- Chat Output ---")
    print(bot.process_input(chat_input))
