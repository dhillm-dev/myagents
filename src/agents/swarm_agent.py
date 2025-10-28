"""
Swarm Agent for AI consensus and signal validation
Uses multiple AI perspectives to validate trading signals
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import httpx
from dataclasses import dataclass
from enum import Enum

from ..config import get_config
from ..bus import get_message_bus, publish_analysis_result

logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Consensus levels"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class SwarmVote:
    """Individual swarm member vote"""
    agent_id: str
    signal: str
    confidence: float
    reasoning: str
    risk_assessment: str
    timestamp: datetime


@dataclass
class ConsensusResult:
    """Swarm consensus result"""
    symbol: str
    consensus: ConsensusLevel
    confidence: float
    votes: List[SwarmVote]
    reasoning: str
    risk_factors: List[str]
    timestamp: datetime


class SwarmAgent:
    """Swarm agent for AI consensus"""
    
    def __init__(self):
        self.config = get_config()
        self.consensus_cache: Dict[str, ConsensusResult] = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Swarm configuration
        self.swarm_size = 5  # Number of AI perspectives
        self.consensus_threshold = 0.6  # Minimum agreement for consensus
        self.min_confidence = 0.7  # Minimum confidence for strong signals
        
        # AI perspectives/roles
        self.ai_perspectives = [
            {
                "id": "technical_analyst",
                "role": "Technical Analysis Expert",
                "focus": "Chart patterns, indicators, and technical signals",
                "bias": "technical"
            },
            {
                "id": "fundamental_analyst", 
                "role": "Fundamental Analysis Expert",
                "focus": "Economic data, company fundamentals, and market conditions",
                "bias": "fundamental"
            },
            {
                "id": "risk_manager",
                "role": "Risk Management Specialist",
                "focus": "Risk assessment, position sizing, and capital preservation",
                "bias": "conservative"
            },
            {
                "id": "momentum_trader",
                "role": "Momentum Trading Expert",
                "focus": "Price momentum, volume analysis, and trend following",
                "bias": "aggressive"
            },
            {
                "id": "contrarian_analyst",
                "role": "Contrarian Analysis Expert",
                "focus": "Market sentiment extremes and reversal opportunities",
                "bias": "contrarian"
            }
        ]
    
    async def get_consensus(
        self,
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        force_refresh: bool = False
    ) -> Optional[ConsensusResult]:
        """Get swarm consensus for a trading signal"""
        try:
            # Check cache first
            if not force_refresh and symbol in self.consensus_cache:
                cached_result = self.consensus_cache[symbol]
                if datetime.now() - cached_result.timestamp < self.cache_duration:
                    return cached_result
            
            # Gather votes from swarm members
            votes = await self._gather_swarm_votes(
                symbol, chart_analysis, sentiment_data, market_data
            )
            
            if not votes:
                logger.warning(f"No votes received for {symbol}")
                return None
            
            # Calculate consensus
            consensus_result = await self._calculate_consensus(symbol, votes)
            
            # Cache result
            self.consensus_cache[symbol] = consensus_result
            
            # Publish consensus result
            await self._publish_consensus(consensus_result)
            
            logger.info(f"Swarm consensus for {symbol}: {consensus_result.consensus.value} (confidence: {consensus_result.confidence:.2f})")
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Failed to get swarm consensus for {symbol}: {e}")
            return None
    
    async def _gather_swarm_votes(
        self,
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> List[SwarmVote]:
        """Gather votes from all swarm members"""
        votes = []
        
        # Create tasks for parallel voting
        vote_tasks = []
        for perspective in self.ai_perspectives:
            task = self._get_perspective_vote(
                perspective, symbol, chart_analysis, sentiment_data, market_data
            )
            vote_tasks.append(task)
        
        # Wait for all votes
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(vote_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get vote from {self.ai_perspectives[i]['id']}: {result}")
            elif result:
                votes.append(result)
        
        return votes
    
    async def _get_perspective_vote(
        self,
        perspective: Dict[str, Any],
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[SwarmVote]:
        """Get vote from a specific AI perspective"""
        try:
            # Build prompt for this perspective
            prompt = await self._build_perspective_prompt(
                perspective, symbol, chart_analysis, sentiment_data, market_data
            )
            
            # Get AI response
            response = await self._query_ai(prompt)
            if not response:
                return None
            
            # Parse response
            vote_data = await self._parse_vote_response(response)
            if not vote_data:
                return None
            
            # Create vote
            vote = SwarmVote(
                agent_id=perspective['id'],
                signal=vote_data['signal'],
                confidence=vote_data['confidence'],
                reasoning=vote_data['reasoning'],
                risk_assessment=vote_data['risk_assessment'],
                timestamp=datetime.now()
            )
            
            return vote
            
        except Exception as e:
            logger.error(f"Failed to get vote from {perspective['id']}: {e}")
            return None
    
    async def _build_perspective_prompt(
        self,
        perspective: Dict[str, Any],
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """Build AI prompt for specific perspective"""
        try:
            # Base context
            context = f"""
You are a {perspective['role']} analyzing {symbol} for trading opportunities.
Your focus is on: {perspective['focus']}
Your analytical bias is: {perspective['bias']}

Current Market Data:
- Symbol: {symbol}
- Current Price: {market_data.get('price', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- 24h Change: {market_data.get('change_24h', 'N/A')}%

Technical Analysis Summary:
- Overall Trend: {chart_analysis.get('trend', 'N/A')}
- RSI: {chart_analysis.get('rsi', 'N/A')}
- EMA Signal: {chart_analysis.get('ema_signal', 'N/A')}
- MACD Signal: {chart_analysis.get('macd_signal', 'N/A')}
- Support Level: {chart_analysis.get('support_level', 'N/A')}
- Resistance Level: {chart_analysis.get('resistance_level', 'N/A')}
- Volatility (ATR): {chart_analysis.get('atr', 'N/A')}
- Pattern Detected: {chart_analysis.get('pattern', 'None')}

Sentiment Analysis:
- Overall Sentiment: {sentiment_data.get('sentiment_score', 'N/A')}/10
- Tradability Score: {sentiment_data.get('tradability_score', 'N/A')}/10
- Key Drivers: {', '.join(sentiment_data.get('key_drivers', []))}
- Market Sentiment: {sentiment_data.get('market_sentiment', 'N/A')}
"""
            
            # Add perspective-specific context
            if perspective['bias'] == 'technical':
                context += f"""
Focus on technical indicators, chart patterns, and price action.
Consider: trend strength, momentum, support/resistance levels, and technical signals.
"""
            elif perspective['bias'] == 'fundamental':
                context += f"""
Focus on fundamental factors, economic conditions, and long-term value.
Consider: market sentiment, economic drivers, and fundamental strength.
"""
            elif perspective['bias'] == 'conservative':
                context += f"""
Focus on risk management and capital preservation.
Consider: downside risk, volatility, and risk-adjusted returns.
"""
            elif perspective['bias'] == 'aggressive':
                context += f"""
Focus on momentum and trend-following opportunities.
Consider: price momentum, volume confirmation, and trend strength.
"""
            elif perspective['bias'] == 'contrarian':
                context += f"""
Focus on sentiment extremes and potential reversals.
Consider: overbought/oversold conditions, sentiment extremes, and reversal patterns.
"""
            
            # Add response format
            context += f"""

Based on your analysis, provide your trading recommendation in the following JSON format:
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed reasoning (2-3 sentences)",
    "risk_assessment": "LOW|MEDIUM|HIGH",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Be specific about your reasoning and consider your analytical bias.
"""
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build perspective prompt: {e}")
            return ""
    
    async def _query_ai(self, prompt: str) -> Optional[str]:
        """Query Perplexity AI for analysis"""
        try:
            async with httpx.AsyncClient(timeout=self.config.pplx_finance.timeout) as client:
                response = await client.post(
                    f"{self.config.pplx_finance.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.pplx_finance.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar-pro",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional trading analyst. Provide concise, actionable analysis in the requested JSON format."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"AI query failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to query AI: {e}")
            return None
    
    async def _parse_vote_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response into vote data"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in AI response")
                return None
            
            json_str = response[start_idx:end_idx]
            vote_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'reasoning', 'risk_assessment']
            for field in required_fields:
                if field not in vote_data:
                    logger.error(f"Missing required field: {field}")
                    return None
            
            # Normalize signal
            signal = vote_data['signal'].upper()
            if signal not in ['BUY', 'SELL', 'HOLD']:
                logger.error(f"Invalid signal: {signal}")
                return None
            
            # Validate confidence
            confidence = float(vote_data['confidence'])
            if not 0 <= confidence <= 1:
                logger.error(f"Invalid confidence: {confidence}")
                return None
            
            # Normalize risk assessment
            risk = vote_data['risk_assessment'].upper()
            if risk not in ['LOW', 'MEDIUM', 'HIGH']:
                logger.error(f"Invalid risk assessment: {risk}")
                return None
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': vote_data['reasoning'],
                'risk_assessment': risk
            }
            
        except Exception as e:
            logger.error(f"Failed to parse vote response: {e}")
            return None
    
    async def _calculate_consensus(self, symbol: str, votes: List[SwarmVote]) -> ConsensusResult:
        """Calculate consensus from swarm votes"""
        try:
            if not votes:
                return ConsensusResult(
                    symbol=symbol,
                    consensus=ConsensusLevel.NEUTRAL,
                    confidence=0.0,
                    votes=[],
                    reasoning="No votes received",
                    risk_factors=[],
                    timestamp=datetime.now()
                )
            
            # Count votes by signal
            vote_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            confidence_sum = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for vote in votes:
                vote_counts[vote.signal] += 1
                confidence_sum[vote.signal] += vote.confidence
            
            # Calculate weighted scores
            total_votes = len(votes)
            buy_score = (vote_counts['BUY'] / total_votes) * (confidence_sum['BUY'] / max(vote_counts['BUY'], 1))
            sell_score = (vote_counts['SELL'] / total_votes) * (confidence_sum['SELL'] / max(vote_counts['SELL'], 1))
            hold_score = (vote_counts['HOLD'] / total_votes) * (confidence_sum['HOLD'] / max(vote_counts['HOLD'], 1))
            
            # Determine consensus
            max_score = max(buy_score, sell_score, hold_score)
            
            if max_score == buy_score:
                if buy_score >= 0.8:
                    consensus = ConsensusLevel.STRONG_BUY
                elif buy_score >= 0.6:
                    consensus = ConsensusLevel.BUY
                else:
                    consensus = ConsensusLevel.WEAK_BUY
            elif max_score == sell_score:
                if sell_score >= 0.8:
                    consensus = ConsensusLevel.STRONG_SELL
                elif sell_score >= 0.6:
                    consensus = ConsensusLevel.SELL
                else:
                    consensus = ConsensusLevel.WEAK_SELL
            else:
                consensus = ConsensusLevel.NEUTRAL
            
            # Calculate overall confidence
            overall_confidence = max_score
            
            # Generate reasoning
            reasoning = await self._generate_consensus_reasoning(votes, consensus)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(votes)
            
            return ConsensusResult(
                symbol=symbol,
                consensus=consensus,
                confidence=overall_confidence,
                votes=votes,
                reasoning=reasoning,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate consensus: {e}")
            return ConsensusResult(
                symbol=symbol,
                consensus=ConsensusLevel.NEUTRAL,
                confidence=0.0,
                votes=votes,
                reasoning=f"Error calculating consensus: {e}",
                risk_factors=[],
                timestamp=datetime.now()
            )
    
    async def _generate_consensus_reasoning(self, votes: List[SwarmVote], consensus: ConsensusLevel) -> str:
        """Generate consensus reasoning from votes"""
        try:
            # Count vote distribution
            buy_votes = [v for v in votes if v.signal == 'BUY']
            sell_votes = [v for v in votes if v.signal == 'SELL']
            hold_votes = [v for v in votes if v.signal == 'HOLD']
            
            reasoning_parts = []
            
            # Add vote distribution
            reasoning_parts.append(f"Swarm consensus: {len(buy_votes)} BUY, {len(sell_votes)} SELL, {len(hold_votes)} HOLD votes")
            
            # Add key reasoning from majority votes
            if consensus.value.startswith('BUY') or consensus.value == 'STRONG_BUY':
                key_reasons = [v.reasoning for v in buy_votes]
            elif consensus.value.startswith('SELL') or consensus.value == 'STRONG_SELL':
                key_reasons = [v.reasoning for v in sell_votes]
            else:
                key_reasons = [v.reasoning for v in votes]
            
            if key_reasons:
                reasoning_parts.append(f"Key factors: {'; '.join(key_reasons[:2])}")
            
            return '. '.join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate consensus reasoning: {e}")
            return "Consensus calculated from swarm votes"
    
    async def _identify_risk_factors(self, votes: List[SwarmVote]) -> List[str]:
        """Identify risk factors from votes"""
        try:
            risk_factors = []
            
            # Count high risk assessments
            high_risk_count = len([v for v in votes if v.risk_assessment == 'HIGH'])
            medium_risk_count = len([v for v in votes if v.risk_assessment == 'MEDIUM'])
            
            if high_risk_count > 0:
                risk_factors.append(f"High risk identified by {high_risk_count} analysts")
            
            if medium_risk_count > len(votes) / 2:
                risk_factors.append("Moderate risk consensus")
            
            # Check for conflicting signals
            signals = set(v.signal for v in votes)
            if len(signals) > 1:
                risk_factors.append("Mixed signals from swarm")
            
            # Check for low confidence
            avg_confidence = sum(v.confidence for v in votes) / len(votes)
            if avg_confidence < 0.6:
                risk_factors.append("Low confidence consensus")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Failed to identify risk factors: {e}")
            return []
    
    async def _publish_consensus(self, consensus_result: ConsensusResult) -> None:
        """Publish consensus result to message bus"""
        try:
            consensus_data = {
                'type': 'swarm_consensus',
                'symbol': consensus_result.symbol,
                'consensus': consensus_result.consensus.value,
                'confidence': consensus_result.confidence,
                'vote_count': len(consensus_result.votes),
                'reasoning': consensus_result.reasoning,
                'risk_factors': consensus_result.risk_factors,
                'timestamp': consensus_result.timestamp.isoformat()
            }
            
            await publish_analysis_result(
                symbol=consensus_result.symbol,
                analysis_type='swarm_consensus',
                data=consensus_data
            )
            
        except Exception as e:
            logger.error(f"Failed to publish consensus: {e}")
    
    async def validate_signal(
        self,
        symbol: str,
        proposed_signal: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Validate a proposed trading signal against swarm consensus"""
        try:
            # Get swarm consensus
            consensus_result = await self.get_consensus(
                symbol, chart_analysis, sentiment_data, market_data
            )
            
            if not consensus_result:
                return False, 0.0, "Failed to get swarm consensus"
            
            # Check if proposed signal aligns with consensus
            proposed_signal = proposed_signal.upper()
            consensus_signal = consensus_result.consensus.value
            
            # Define alignment rules
            alignment_map = {
                'BUY': ['STRONG_BUY', 'BUY', 'WEAK_BUY'],
                'SELL': ['STRONG_SELL', 'SELL', 'WEAK_SELL'],
                'HOLD': ['NEUTRAL']
            }
            
            is_aligned = consensus_signal in alignment_map.get(proposed_signal, [])
            
            # Calculate validation confidence
            validation_confidence = consensus_result.confidence if is_aligned else 1 - consensus_result.confidence
            
            # Generate validation message
            if is_aligned:
                message = f"Signal validated by swarm consensus: {consensus_signal} (confidence: {consensus_result.confidence:.2f})"
            else:
                message = f"Signal conflicts with swarm consensus: {consensus_signal} vs {proposed_signal}"
            
            return is_aligned, validation_confidence, message
            
        except Exception as e:
            logger.error(f"Failed to validate signal: {e}")
            return False, 0.0, f"Validation error: {e}"
    
    async def get_swarm_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get swarm consensus summary for multiple symbols"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'consensus_distribution': {
                    'STRONG_BUY': 0,
                    'BUY': 0,
                    'WEAK_BUY': 0,
                    'NEUTRAL': 0,
                    'WEAK_SELL': 0,
                    'SELL': 0,
                    'STRONG_SELL': 0
                },
                'average_confidence': 0.0,
                'high_confidence_signals': [],
                'risk_alerts': []
            }
            
            total_confidence = 0
            consensus_count = 0
            
            for symbol in symbols:
                if symbol in self.consensus_cache:
                    consensus = self.consensus_cache[symbol]
                    
                    # Update distribution
                    summary['consensus_distribution'][consensus.consensus.value] += 1
                    
                    # Update confidence
                    total_confidence += consensus.confidence
                    consensus_count += 1
                    
                    # Check for high confidence signals
                    if consensus.confidence >= self.min_confidence:
                        summary['high_confidence_signals'].append({
                            'symbol': symbol,
                            'consensus': consensus.consensus.value,
                            'confidence': consensus.confidence
                        })
                    
                    # Check for risk alerts
                    if consensus.risk_factors:
                        summary['risk_alerts'].append({
                            'symbol': symbol,
                            'risk_factors': consensus.risk_factors
                        })
            
            # Calculate average confidence
            if consensus_count > 0:
                summary['average_confidence'] = total_confidence / consensus_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get swarm summary: {e}")
            return {}


# Global swarm agent instance
_swarm_agent: Optional[SwarmAgent] = None


async def get_swarm_agent() -> SwarmAgent:
    """Get the global swarm agent"""
    global _swarm_agent
    
    if _swarm_agent is None:
        _swarm_agent = SwarmAgent()
    
    return _swarm_agent