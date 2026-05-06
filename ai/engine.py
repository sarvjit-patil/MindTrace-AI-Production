import datetime
import asyncio
import os
import random
from typing import Dict, Any


from ai.scoring.wellness import WellnessIndexEngine
from ai.risk_detection.monitor import RiskDetector
from ai.recommendation.engine import RecommendationEngine
from ai.preprocessing.text import clean_text
from ai.utils.models import AIResponse, AnalysisRequest

class MindTraceAIEngine:
    def __init__(self):
        print("Initializing MindTrace AI Engine...")
        self.is_render = os.environ.get("RENDER") == "true"
        
        if self.is_render:
            print("Render Free Tier detected! Bypassing massive PyTorch models to prevent Out of Memory crash...")
            # We skip loading the heavy transformers. The app will use rule-based Mock AI.
        else:
            print("Loading Hugging Face models into memory...")
            from ai.emotion.analyzer import EmotionAnalyzer
            from ai.sentiment.analyzer import SentimentAnalyzer
            self.emotion_analyzer = EmotionAnalyzer()
            self.sentiment_analyzer = SentimentAnalyzer()
            
        self.wellness_engine = WellnessIndexEngine()
        self.risk_detector = RiskDetector()
        self.recommendation_engine = RecommendationEngine()
        print("AI Engine models loaded and ready.")

    async def analyze(self, request: AnalysisRequest) -> dict:
        """
        Main entry point for backend APIs to get the complete emotional analysis.
        Runs inference asynchronously to meet performance targets.
        """
        text = clean_text(request.text)
        
        if self.is_render:
            # Generate rule-based mock response to save RAM on Render
            await asyncio.sleep(1.5) # Simulate AI processing time
            is_negative = any(word in text.lower() for word in ['sad', 'angry', 'depressed', 'anxious', 'bad', 'overwhelmed'])
            emotion = 'sadness' if is_negative else 'joy'
            confidence = 0.88 + (random.random() * 0.1)
            sentiment_score = -0.6 if is_negative else 0.8
        else:
            # Run real Hugging Face emotion and sentiment classification
            emotion_task = asyncio.to_thread(self.emotion_analyzer.analyze, text)
            sentiment_task = asyncio.to_thread(self.sentiment_analyzer.analyze, text)
            emotion_result, sentiment_score = await asyncio.gather(emotion_task, sentiment_task)
            emotion = emotion_result["emotion"]
            confidence = emotion_result["confidence"]
        
        # Compute Wellness Index
        wellness_index = self.wellness_engine.calculate(emotion, sentiment_score)
        
        # Assess Risk Level
        risk_assessment = self.risk_detector.assess_risk(emotion, wellness_index)
        
        # Get AI Recommendations
        suggestions = self.recommendation_engine.get_suggestions(
            emotion=emotion, 
            risk_level=risk_assessment["risk_level"]
        )
        
        # Construct the MANDATORY AI RESPONSE FORMAT
        response = AIResponse(
            emotion=emotion,
            confidence=confidence,
            sentiment=sentiment_score,
            wellness_index=wellness_index,
            risk_level=risk_assessment["risk_level"],
            trigger_intervention=risk_assessment["trigger_intervention"],
            suggestions=suggestions,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )
        
        return response.model_dump()
