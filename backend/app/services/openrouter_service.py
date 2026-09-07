import io
import base64
import json
import logging
import requests
import certifi
from PIL import Image
from backend.app.config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_URL,
    SUPPORTED_BREED_NAMES
)

logger = logging.getLogger(__name__)

class OpenRouterAIService:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.url = OPENROUTER_URL

    def analyze_breed(self, image: Image.Image) -> dict | None:
        """
        Sends animal image to OpenRouter Vision AI to classify breed strictly within supported list.
        Returns dict {"ai_breed": str, "ai_confidence": float, "ai_reasoning": str} or None on failure.
        """
        if not self.api_key:
            logger.info("OPENROUTER_API_KEY is not configured. OpenRouter AI analysis skipped.")
            return None

        try:
            # Convert PIL image to base64 JPEG string
            buffered = io.BytesIO()
            image.convert("RGB").save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_b64}"

            system_prompt = (
                "You are an expert livestock veterinary researcher specializing in Indian Cattle and Buffalo breeds. "
                "Analyze the provided image and classify the animal breed strictly into EXACTLY ONE of the following 9 breeds:\n"
                f"{', '.join(SUPPORTED_BREED_NAMES)}.\n\n"
                "Return ONLY a raw valid JSON object with these exact keys:\n"
                "{\n"
                '  "breed": "<Exact breed name from the list>",\n'
                '  "confidence": <float probability between 0.0 and 1.0>,\n'
                '  "reasoning": "<Short 1-2 sentence visual justification based on horns, ears, hump, coat color, or body structure>"\n'
                "}\n"
                "Do NOT use markdown code blocks or extra commentary outside the JSON."
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-breed-identifier.local",
                "X-Title": "Cattle & Buffalo Breed Vision AI"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }

            # Attempt HTTPS request with certifi CA bundle first, fallback to verify=False if Windows SSL cert fails
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=25, verify=certifi.where())
            except requests.exceptions.SSLError:
                logger.warning("SSL Certificate verification failed; retrying OpenRouter API request without SSL verification.")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.post(self.url, headers=headers, json=payload, timeout=25, verify=False)
            
            if response.status_code != 200:
                logger.warning(f"OpenRouter API error response (Status {response.status_code}): {response.text[:200]}")
                return None

            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                return None

            content = choices[0].get("message", {}).get("content", "").strip()
            
            # Clean markdown code block formatting if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            raw_breed = str(parsed.get("breed", "")).strip()
            confidence = float(parsed.get("confidence", 0.75))
            confidence = max(0.0, min(1.0, round(confidence, 4)))
            reasoning = str(parsed.get("reasoning", "Visual characteristics match breed features.")).strip()

            # Enforce strict breed name matching against supported list
            matched_breed = None
            for sup in SUPPORTED_BREED_NAMES:
                if sup.lower() == raw_breed.lower():
                    matched_breed = sup
                    break

            if not matched_breed:
                for sup in SUPPORTED_BREED_NAMES:
                    if sup.lower() in raw_breed.lower() or raw_breed.lower() in sup.lower():
                        matched_breed = sup
                        break

            if not matched_breed:
                logger.warning(f"OpenRouter returned unsupported breed '{raw_breed}'. Discarding AI result.")
                return None

            return {
                "ai_breed": matched_breed,
                "ai_confidence": confidence,
                "ai_reasoning": reasoning
            }

        except Exception as e:
            logger.error(f"OpenRouter AI analysis failed: {str(e)}")
            return None

openrouter_service = OpenRouterAIService()
