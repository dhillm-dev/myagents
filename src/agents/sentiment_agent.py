"""
Sentiment Analysis Agent using Perplexity Finance API
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import httpx

from ..config import get_config
from ..bus import publish_analysis_result

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    """Sentiment analysis agent using Perplexity Finance API"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)  # Cache sentiment for 15 minutes
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.perplexity.timeout),
                headers={
                    'Authorization': f'Bearer {self.config.perplexity.api_key}',
                    'Content-Type': 'application/json'
                }
            )
        return self.client
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def analyze_sentiment(self, symbol: str, context: str = "trading") -> Optional[Dict[str, Any]]:
        """Analyze sentiment for a symbol using Perplexity Finance API"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{context}"
            if self._is_cached(cache_key):
                return self.sentiment_cache[cache_key]
            
            # Get sentiment analysis from Perplexity
            sentiment_data = await self._get_perplexity_sentiment(symbol, context)
            if not sentiment_data:
                return None
            
            # Process and enhance the sentiment data
            processed_sentiment = await self._process_sentiment_data(sentiment_data, symbol)
            
            # Cache the result
            self.sentiment_cache[cache_key] = processed_sentiment
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            # Publish analysis result
            await publish_analysis_result(
                symbol=symbol,
                analysis_type='sentiment_analysis',
                result=processed_sentiment,
                confidence=processed_sentiment.get('confidence', 0.5)
            )
            
            return processed_sentiment
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment for {symbol}: {e}")
            return None
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if sentiment is cached and not expired"""
        if cache_key not in self.sentiment_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    async def _get_perplexity_sentiment(self, symbol: str, context: str) -> Optional[Dict[str, Any]]:
        """Get sentiment analysis from Perplexity Finance API"""
        try:
            client = await self._get_client()
            
            # Construct the prompt for sentiment analysis
            prompt = self._build_sentiment_prompt(symbol, context)
            
            payload = {
                "model": self.config.perplexity.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in market sentiment analysis. Provide objective, data-driven sentiment analysis based on recent market data, news, and financial indicators."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.9,
                "return_citations": True,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day",
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            
            response = await client.post(
                f"{self.config.perplexity.base_url}/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_perplexity_response(data)
            else:
                logger.error(f"Perplexity API error {response.status_code}: {response.text}")
                return None
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling Perplexity API for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error calling Perplexity API for {symbol}: {e}")
            return None
    
    def _build_sentiment_prompt(self, symbol: str, context: str) -> str:
        """Build prompt for Perplexity sentiment analysis"""
        base_prompt = f"""
        Analyze the current market sentiment and tradability for {symbol} in the context of {context}.
        
        Please provide a comprehensive analysis including:
        
        1. **Overall Sentiment Score** (0-100, where 0 is extremely bearish, 50 is neutral, 100 is extremely bullish)
        
        2. **Key Sentiment Drivers** (list the top 3-5 factors influencing current sentiment)
        
        3. **Recent News Impact** (how recent news/events are affecting sentiment)
        
        4. **Market Momentum** (current price action and volume trends)
        
        5. **Institutional Activity** (any notable institutional buying/selling)
        
        6. **Technical Sentiment** (how technical indicators align with fundamental sentiment)
        
        7. **Risk Factors** (key risks that could change sentiment)
        
        8. **Tradability Score** (0-100, considering liquidity, volatility, and market conditions)
        
        9. **Time Horizon** (short-term vs long-term sentiment outlook)
        
        10. **Confidence Level** (0-100, how confident you are in this analysis)
        
        Please format your response as a structured analysis with clear sections and specific numerical scores where requested.
        Focus on recent data (last 24-48 hours) and current market conditions.
        """
        
        return base_prompt.strip()
    
    def _parse_perplexity_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Perplexity API response"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                return {}
            
            content = response_data['choices'][0]['message']['content']
            citations = response_data.get('citations', [])
            
            # Extract structured data from the response
            sentiment_data = {
                'raw_content': content,
                'citations': citations,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to extract numerical scores using simple parsing
            sentiment_data.update(self._extract_scores_from_content(content))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error parsing Perplexity response: {e}")
            return {}
    
    def _extract_scores_from_content(self, content: str) -> Dict[str, Any]:
        """Extract numerical scores from content using simple text parsing"""
        try:
            import re
            
            scores = {}
            
            # Extract sentiment score
            sentiment_match = re.search(r'sentiment.*?score.*?(\d+)', content, re.IGNORECASE)
            if sentiment_match:
                scores['sentiment_score'] = int(sentiment_match.group(1))
            
            # Extract tradability score
            tradability_match = re.search(r'tradability.*?score.*?(\d+)', content, re.IGNORECASE)
            if tradability_match:
                scores['tradability_score'] = int(tradability_match.group(1))
            
            # Extract confidence level
            confidence_match = re.search(r'confidence.*?level.*?(\d+)', content, re.IGNORECASE)
            if confidence_match:
                scores['confidence_level'] = int(confidence_match.group(1))
            
            # Extract key sentiment drivers
            drivers = []
            drivers_section = re.search(r'key sentiment drivers.*?:(.*?)(?=\n\d+\.|\n[A-Z]|\Z)', content, re.IGNORECASE | re.DOTALL)
            if drivers_section:
                driver_text = drivers_section.group(1)
                # Simple extraction of bullet points or numbered items
                driver_matches = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', driver_text)
                drivers = [driver.strip() for driver in driver_matches[:5]]  # Top 5
            
            scores['sentiment_drivers'] = drivers
            
            return scores
            
        except Exception as e:
            logger.error(f"Error extracting scores from content: {e}")
            return {}
    
    async def _process_sentiment_data(self, sentiment_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process and enhance sentiment data"""
        try:
            processed = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'source': 'perplexity_finance',
                'raw_data': sentiment_data
            }
            
            # Extract and normalize scores
            sentiment_score = sentiment_data.get('sentiment_score', 50)
            tradability_score = sentiment_data.get('tradability_score', 50)
            confidence_level = sentiment_data.get('confidence_level', 50)
            
            # Normalize scores to 0-1 range
            processed['sentiment'] = {
                'score': sentiment_score / 100.0,
                'direction': self._get_sentiment_direction(sentiment_score),
                'strength': self._get_sentiment_strength(sentiment_score)
            }
            
            processed['tradability'] = {
                'score': tradability_score / 100.0,
                'rating': self._get_tradability_rating(tradability_score)
            }
            
            processed['confidence'] = confidence_level / 100.0
            
            # Extract sentiment drivers
            processed['drivers'] = sentiment_data.get('sentiment_drivers', [])
            
            # Calculate overall recommendation
            processed['recommendation'] = self._get_sentiment_recommendation(
                sentiment_score, tradability_score, confidence_level
            )
            
            # Risk assessment
            processed['risk_assessment'] = self._assess_sentiment_risk(
                sentiment_score, confidence_level
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing sentiment data: {e}")
            return {}
    
    def _get_sentiment_direction(self, score: int) -> str:
        """Get sentiment direction from score"""
        if score >= 60:
            return 'bullish'
        elif score <= 40:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_sentiment_strength(self, score: int) -> str:
        """Get sentiment strength from score"""
        if score >= 80 or score <= 20:
            return 'strong'
        elif score >= 65 or score <= 35:
            return 'moderate'
        else:
            return 'weak'
    
    def _get_tradability_rating(self, score: int) -> str:
        """Get tradability rating from score"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        else:
            return 'poor'
    
    def _get_sentiment_recommendation(self, sentiment: int, tradability: int, confidence: int) -> str:
        """Get overall recommendation based on sentiment analysis"""
        if confidence < 40:
            return 'INSUFFICIENT_DATA'
        
        if tradability < 30:
            return 'AVOID'
        
        if sentiment >= 70 and tradability >= 60 and confidence >= 60:
            return 'STRONG_BUY'
        elif sentiment >= 60 and tradability >= 50:
            return 'BUY'
        elif sentiment <= 30 and tradability >= 60 and confidence >= 60:
            return 'STRONG_SELL'
        elif sentiment <= 40 and tradability >= 50:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _assess_sentiment_risk(self, sentiment: int, confidence: int) -> Dict[str, Any]:
        """Assess risk based on sentiment analysis"""
        risk_level = 'medium'
        risk_factors = []
        
        if confidence < 50:
            risk_level = 'high'
            risk_factors.append('Low confidence in analysis')
        
        if sentiment > 80:
            risk_factors.append('Extremely bullish sentiment - potential for reversal')
        elif sentiment < 20:
            risk_factors.append('Extremely bearish sentiment - potential for reversal')
        
        if 45 <= sentiment <= 55:
            risk_factors.append('Neutral sentiment - unclear direction')
        
        # Determine overall risk level
        if len(risk_factors) >= 2:
            risk_level = 'high'
        elif len(risk_factors) == 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'level': risk_level,
            'factors': risk_factors,
            'score': len(risk_factors) / 3.0  # Normalize to 0-1
        }
    
    async def get_tradability_score(self, symbol: str) -> Optional[float]:
        """Get tradability score for a symbol"""
        try:
            sentiment_data = await self.analyze_sentiment(symbol, "tradability")
            if sentiment_data and 'tradability' in sentiment_data:
                return sentiment_data['tradability']['score']
            return None
        except Exception as e:
            logger.error(f"Error getting tradability score for {symbol}: {e}")
            return None
    
    async def get_market_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get overall market sentiment summary for multiple symbols"""
        try:
            sentiment_results = []
            
            # Analyze sentiment for each symbol
            for symbol in symbols:
                sentiment = await self.analyze_sentiment(symbol)
                if sentiment:
                    sentiment_results.append(sentiment)
            
            if not sentiment_results:
                return {}
            
            # Calculate aggregate metrics
            avg_sentiment = sum(s['sentiment']['score'] for s in sentiment_results) / len(sentiment_results)
            avg_tradability = sum(s['tradability']['score'] for s in sentiment_results) / len(sentiment_results)
            avg_confidence = sum(s['confidence'] for s in sentiment_results) / len(sentiment_results)
            
            # Count recommendations
            recommendations = [s['recommendation'] for s in sentiment_results]
            rec_counts = {}
            for rec in recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(sentiment_results),
                'average_sentiment': avg_sentiment,
                'average_tradability': avg_tradability,
                'average_confidence': avg_confidence,
                'market_direction': self._get_sentiment_direction(int(avg_sentiment * 100)),
                'recommendation_distribution': rec_counts,
                'overall_market_sentiment': 'bullish' if avg_sentiment > 0.6 else 'bearish' if avg_sentiment < 0.4 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error getting market sentiment summary: {e}")
            return {}
    
    async def start_continuous_sentiment_analysis(self, symbols: List[str], interval: int = 900) -> None:
        """Start continuous sentiment analysis for multiple symbols (15-minute default)"""
        self.running = True
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._sentiment_worker(symbol, interval))
            tasks.append(task)
        
        logger.info(f"Started continuous sentiment analysis for {len(symbols)} symbols")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in continuous sentiment analysis: {e}")
    
    async def _sentiment_worker(self, symbol: str, interval: int) -> None:
        """Worker for continuous sentiment analysis"""
        while self.running:
            try:
                await self.analyze_sentiment(symbol)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                await asyncio.sleep(interval * 2)
    
    async def stop_continuous_analysis(self) -> None:
        """Stop continuous sentiment analysis"""
        self.running = False
        await self.close()
        logger.info("Stopped continuous sentiment analysis")


# Global sentiment agent instance
_sentiment_agent: Optional[SentimentAnalysisAgent] = None


async def get_sentiment_agent() -> SentimentAnalysisAgent:
    """Get the global sentiment analysis agent"""
    global _sentiment_agent
    
    if _sentiment_agent is None:
        _sentiment_agent = SentimentAnalysisAgent()
    
    return _sentiment_agent


async def close_sentiment_agent():
    """Close the global sentiment agent"""
    global _sentiment_agent
    
    if _sentiment_agent:
        await _sentiment_agent.close()
        _sentiment_agent = None