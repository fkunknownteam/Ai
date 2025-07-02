import asyncio
import json
import os
import re
import requests
import sqlite3
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime.datetime
    metadata: Dict[str, Any] = None

class JarvisAI:
    def __init__(self):
        self.conversation_history = []
        self.memory_db = "jarvis_memory.db"
        self.plugins = {}
        self.capabilities = {
            "text_generation": True,
            "web_search": True,
            "code_execution": True,
            "file_operations": True,
            "image_processing": True,
            "data_analysis": True,
            "memory_storage": True,
            "reasoning": True
        }
        self.initialize_memory()
        self.load_plugins()
        
    def initialize_memory(self):
        """Initialize SQLite database for persistent memory"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                information TEXT,
                source TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_to_memory(self, message: Message):
        """Save conversation to persistent memory"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (timestamp, role, content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            message.timestamp.isoformat(),
            message.role,
            message.content,
            json.dumps(message.metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
    def retrieve_memory(self, query: str, limit: int = 10) -> List[Message]:
        """Retrieve relevant memories based on query"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, role, content, metadata
            FROM conversations
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        memories = []
        for row in results:
            timestamp = datetime.datetime.fromisoformat(row[0])
            metadata = json.loads(row[3]) if row[3] else {}
            memories.append(Message(row[1], row[2], timestamp, metadata))
            
        return memories
        
    def web_search(self, query: str) -> Dict[str, Any]:
        """Simulate web search functionality"""
        # In a real implementation, you'd use APIs like Google Search, Bing, etc.
        search_results = {
            "query": query,
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com",
                    "snippet": f"This is a simulated search result for '{query}'. In a real implementation, this would contain actual web search results."
                }
            ],
            "timestamp": datetime.datetime.now().isoformat()
        }
        return search_results
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        words = text.lower().split()
        positive_score = sum(1 for word in words if word in positive_words)
        negative_score = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        positive_ratio = positive_score / total_words
        negative_ratio = negative_score / total_words
        neutral_ratio = 1 - positive_ratio - negative_ratio
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": max(0, neutral_ratio)
        }
        
    def process_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Process and analyze code"""
        if language.lower() == "python":
            try:
                # Basic Python code analysis
                lines = code.split('\n')
                functions = [line for line in lines if line.strip().startswith('def ')]
                classes = [line for line in lines if line.strip().startswith('class ')]
                imports = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                
                return {
                    "language": language,
                    "lines_of_code": len(lines),
                    "functions": len(functions),
                    "classes": len(classes),
                    "imports": len(imports),
                    "complexity": "medium" if len(lines) > 50 else "low"
                }
            except Exception as e:
                return {"error": str(e)}
        
        return {"message": f"Code analysis for {language} not implemented yet"}
        
    def generate_response(self, user_input: str) -> str:
        """Generate intelligent response based on user input"""
        user_message = Message("user", user_input, datetime.datetime.now())
        self.conversation_history.append(user_message)
        self.save_to_memory(user_message)
        
        # Analyze user input
        sentiment = self.analyze_sentiment(user_input)
        
        # Check for specific commands or patterns
        response = self.process_user_intent(user_input, sentiment)
        
        # Create assistant response
        assistant_message = Message("assistant", response, datetime.datetime.now(), 
                                  {"sentiment_analysis": sentiment})
        self.conversation_history.append(assistant_message)
        self.save_to_memory(assistant_message)
        
        return response
        
    def process_user_intent(self, user_input: str, sentiment: Dict[str, float]) -> str:
        """Process user intent and generate appropriate response"""
        input_lower = user_input.lower()
        
        # Greeting patterns
        if any(greeting in input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm Jarvis, your advanced AI assistant. How can I help you today? I can assist with coding, data analysis, web searches, file operations, and much more!"
            
        # Search requests
        elif any(search_term in input_lower for search_term in ['search', 'find', 'look up', 'google']):
            query = re.sub(r'(search|find|look up|google)\s+(for\s+)?', '', input_lower)
            search_results = self.web_search(query)
            return f"I found some information about '{query}':\n\n{search_results['results'][0]['snippet']}\n\nWould you like me to search for more specific information?"
            
        # Code-related requests
        elif any(code_term in input_lower for code_term in ['code', 'program', 'script', 'function', 'algorithm']):
            return "I'd be happy to help with coding! I can:\n• Write code in multiple languages\n• Debug and optimize existing code\n• Explain programming concepts\n• Create algorithms and data structures\n\nWhat specific coding task would you like assistance with?"
            
        # Data analysis requests
        elif any(data_term in input_lower for data_term in ['data', 'analyze', 'statistics', 'chart', 'graph']):
            return "I can help with data analysis! I can:\n• Process CSV and JSON files\n• Create visualizations and charts\n• Perform statistical analysis\n• Generate insights from data\n\nPlease share your data or describe what analysis you need."
            
        # File operations
        elif any(file_term in input_lower for file_term in ['file', 'document', 'save', 'load', 'read', 'write']):
            return "I can handle various file operations:\n• Read and write text files\n• Process CSV, JSON, and other formats\n• Image processing and manipulation\n• File organization and management\n\nWhat file operation would you like me to perform?"
            
        # Memory and learning
        elif any(memory_term in input_lower for memory_term in ['remember', 'recall', 'memory', 'learn', 'forget']):
            if 'remember' in input_lower:
                return "I'll remember our conversation! I have persistent memory that allows me to recall previous interactions and learn from them. What would you like me to remember specifically?"
            else:
                memories = self.retrieve_memory(user_input, 5)
                if memories:
                    return f"I found {len(memories)} related memories from our previous conversations. Would you like me to share them?"
                else:
                    return "I don't have any specific memories related to that topic yet, but I'm always learning from our interactions!"
                    
        # Capabilities inquiry
        elif any(capability_term in input_lower for capability_term in ['what can you do', 'capabilities', 'features', 'help']):
            return self.get_capabilities_summary()
            
        # Sentiment-based responses
        elif sentiment['negative'] > 0.3:
            return "I sense you might be frustrated or facing a challenge. I'm here to help! Please let me know what specific issue you're dealing with, and I'll do my best to assist you."
            
        elif sentiment['positive'] > 0.3:
            return "I'm glad you're in a positive mood! How can I help make your day even better? I'm ready to assist with any task you have in mind."
            
        # Default intelligent response
        else:
            return self.generate_contextual_response(user_input)
            
    def generate_contextual_response(self, user_input: str) -> str:
        """Generate contextual response based on conversation history"""
        # Analyze recent conversation context
        recent_messages = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        
        context_topics = []
        for msg in recent_messages:
            if msg.role == "user":
                # Extract key topics from recent messages
                words = msg.content.lower().split()
                context_topics.extend([word for word in words if len(word) > 4])
        
        # Generate response based on context
        if context_topics:
            return f"Based on our conversation, I understand you're interested in topics like {', '.join(set(context_topics[:3]))}. Could you provide more specific details about what you'd like to know or accomplish?"
        else:
            return "I'm here to help with a wide range of tasks! Whether you need assistance with coding, data analysis, research, file operations, or just want to have a conversation, I'm ready to assist. What would you like to work on?"
            
    def get_capabilities_summary(self) -> str:
        """Return a summary of Jarvis capabilities"""
        return """
🤖 **Jarvis AI - Advanced Capabilities**

**Core Features:**
• 💬 Natural Language Processing & Conversation
• 🔍 Web Search & Information Retrieval
• 💾 Persistent Memory & Learning
• 🧠 Contextual Understanding & Reasoning

**Technical Capabilities:**
• 💻 Code Generation & Analysis (Python, JavaScript, etc.)
• 📊 Data Analysis & Visualization
• 📁 File Operations & Processing
• 🖼️ Image Processing & Manipulation
• 📈 Statistical Analysis & Machine Learning

**Advanced Features:**
• 🎯 Sentiment Analysis
• 🔗 Plugin System (Extensible)
• 📚 Knowledge Base Management
• 🕒 Conversation History & Context
• 🎨 Creative Problem Solving

**Specialized Tasks:**
• Research & Information Synthesis
• Code Debugging & Optimization
• Data Cleaning & Transformation
• Report Generation
• Educational Explanations

How can I assist you today?
        """
        
    def load_plugins(self):
        """Load additional plugins for extended functionality"""
        # Placeholder for plugin system
        self.plugins = {
            "weather": self.get_weather,
            "calculator": self.calculate,
            "translator": self.translate_text
        }
        
    def get_weather(self, location: str) -> str:
        """Simulated weather plugin"""
        return f"Weather in {location}: Partly cloudy, 72°F (22°C). This is a simulated response - in a real implementation, this would connect to a weather API."
        
    def calculate(self, expression: str) -> str:
        """Safe calculator plugin"""
        try:
            # Basic safety check - only allow numbers and basic operators
            if re.match(r'^[0-9+\-*/().\s]+$', expression):
                result = eval(expression)
                return f"Result: {result}"
            else:
                return "Invalid expression. Please use only numbers and basic operators (+, -, *, /, parentheses)."
        except Exception as e:
            return f"Calculation error: {str(e)}"
            
    def translate_text(self, text: str, target_language: str = "spanish") -> str:
        """Simulated translation plugin"""
        translations = {
            "hello": {"spanish": "hola", "french": "bonjour", "german": "hallo"},
            "goodbye": {"spanish": "adiós", "french": "au revoir", "german": "auf wiedersehen"},
            "thank you": {"spanish": "gracias", "french": "merci", "german": "danke"}
        }
        
        text_lower = text.lower()
        if text_lower in translations and target_language in translations[text_lower]:
            return f"'{text}' in {target_language}: {translations[text_lower][target_language]}"
        else:
            return f"Translation for '{text}' to {target_language} not available in this demo. In a real implementation, this would use translation APIs."

# Main execution
def main():
    jarvis = JarvisAI()
    
    print("🤖 Jarvis AI Assistant Initialized!")
    print("=" * 50)
    print(jarvis.get_capabilities_summary())
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Jarvis: Goodbye! It was great assisting you today.")
                break
                
            if not user_input:
                continue
                
            response = jarvis.generate_response(user_input)
            print(f"Jarvis: {response}\n")
            
        except KeyboardInterrupt:
            print("\nJarvis: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Jarvis: I encountered an error: {str(e)}")

if __name__ == "__main__":
    main()
