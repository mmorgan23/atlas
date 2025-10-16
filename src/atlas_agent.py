"""
Atlas AI Agent - Core Agent System
Level 1: Basic Agent Features
"""
import os
import json
import re
import csv
import string
from difflib import get_close_matches
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import psycopg2

# For LLM Integration (we'll start with a simple approach, then add OpenAI)
import openai
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ConversationContext:
    """Store conversation context and memory"""
    user_id: str = "default"
    conversation_history: List[Dict[str, str]] = None
    current_task: Optional[str] = None
    user_preferences: Dict[str, Any] = None
    pending_actions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.pending_actions is None:
            self.pending_actions = []

class AgentTools:
    """Tool registry for agent actions"""
    
    def __init__(self, db_connection):
        self.db_conn = db_connection
        # Register available tools
        self.tools = {
            'get_customers': self.get_customers,
            'add_customer': self.add_customer,
            'update_customer': self.update_customer,
            'delete_customer': self.delete_customer,
            'search_customers': self.search_customers,
            'search_customer': self.search_customer,
        }
    
    def search_customer(self, name: str) -> Dict[str, Any]:
        """Search for customer by name (exact match only)"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            tokens = [t for t in name.strip().split() if t]
            name_parts = [t.lower() for t in tokens]
            results: list[dict] = []

            if len(name_parts) == 1:
                # Single token: match exact first or last name only
                tok = name_parts[0]
                query = (
                    "SELECT * FROM customers "
                    "WHERE LOWER(first_name) = LOWER(%s) OR LOWER(last_name) = LOWER(%s) "
                    "ORDER BY id DESC LIMIT 50"
                )
                print(f"[DEBUG] Single-token exact search: query={query}, params=({tok}, {tok}), tokens={name_parts}")
                cursor.execute(query, (tok, tok))
                results = cursor.fetchall()
            elif len(name_parts) >= 2:
                # Two+ tokens: try exact first+last, then reversed exact
                first = name_parts[0]
                last = " ".join(name_parts[1:])
                print(f"[DEBUG] Multi-token exact search: first='{first}', last='{last}', tokens={name_parts}")
                cursor.execute(
                    "SELECT * FROM customers WHERE LOWER(first_name) = LOWER(%s) AND LOWER(last_name) = LOWER(%s) ORDER BY id DESC",
                    (first, last)
                )
                results = cursor.fetchall()

                # If none, try reversed (user may input Last First)
                if not results and len(name_parts) == 2:
                    rev_first, rev_last = last, first
                    print(f"[DEBUG] Reversed exact search: SELECT * FROM customers WHERE first_name = %s AND last_name = %s ORDER BY id DESC, params=({rev_first}, {rev_last})")
                    cursor.execute(
                        "SELECT * FROM customers WHERE LOWER(first_name) = LOWER(%s) AND LOWER(last_name) = LOWER(%s) ORDER BY id DESC",
                        (rev_first, rev_last)
                    )
                    results = cursor.fetchall()

            cursor.close()

            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    def add_customer(self, name: str, email: str, phone: str = "") -> Dict[str, Any]:
        """Add a new customer to the database"""
        try:
            # Split name into first and last name
            name_parts = name.strip().split(' ', 1)
            first_name = name_parts[0].strip().capitalize() if name_parts and name_parts[0] else ""
            last_name = name_parts[1].strip().capitalize() if len(name_parts) > 1 else ""
            
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            query = """
                INSERT INTO customers (first_name, last_name, email, phone)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (first_name, last_name, email, phone))
            self.db_conn.commit()
            customer_id = cursor.lastrowid
            cursor.close()
            
            pretty_name = f"{first_name} {last_name}".strip()
            return {
                "success": True,
                "message": f"Customer {pretty_name} added successfully",
                "customer_id": customer_id
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add customer: {str(e)}"
            }
    
    def update_customer(self, customer_id: int, **kwargs) -> Dict[str, Any]:
        """Update customer information"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Build dynamic update query
            update_fields = []
            values = []
            for field, value in kwargs.items():
                if field in ['first_name', 'last_name', 'email', 'phone']:
                    update_fields.append(f"{field} = %s")
                    values.append(value)
            
            if not update_fields:
                return {"success": False, "message": "No valid fields to update"}
            
            query = f"UPDATE customers SET {', '.join(update_fields)} WHERE id = %s"
            values.append(customer_id)
            
            cursor.execute(query, values)
            self.db_conn.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            
            if affected_rows > 0:
                return {"success": True, "message": f"Customer {customer_id} updated successfully"}
            else:
                return {"success": False, "message": f"Customer {customer_id} not found"}
                
        except Exception as e:
            return {"success": False, "message": f"Failed to update customer: {str(e)}"}
    
    def delete_customer(self, customer_id: int) -> Dict[str, Any]:
        """Delete a customer from the database"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            query = "DELETE FROM customers WHERE id = %s"
            cursor.execute(query, (customer_id,))
            self.db_conn.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            
            if affected_rows > 0:
                return {"success": True, "message": f"Customer {customer_id} deleted successfully"}
            else:
                return {"success": False, "message": f"Customer {customer_id} not found"}
                
        except Exception as e:
            return {"success": False, "message": f"Failed to delete customer: {str(e)}"}
    
    def search_customers(self, query: str) -> Dict[str, Any]:
        """Search customers by name or email"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            search_query = """
                SELECT * FROM customers 
                WHERE first_name LIKE %s OR last_name LIKE %s OR email LIKE %s 
                ORDER BY first_name, last_name
            """
            search_term = f"%{query}%"
            cursor.execute(search_query, (search_term, search_term, search_term))
            customers = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "customers": customers,
                "count": len(customers)
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to search customers: {str(e)}"}
    
    def get_customers(self, limit: int = 10) -> Dict[str, Any]:
        """Get list of customers from the database"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            query = "SELECT * FROM customers ORDER BY id DESC LIMIT %s"
            cursor.execute(query, (limit,))
            customers = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "results": customers,
                "customers": customers,
                "count": len(customers)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to retrieve customers: {str(e)}"
            }
    
    def get_customer_stats(self) -> Dict[str, Any]:
        """Get customer database statistics"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            total = cursor.fetchone()[0]
            cursor.close()
            
            return {
                "success": True,
                "total_customers": int(total),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def schedule_reminder(self, customer_name: str, reminder_text: str) -> Dict[str, Any]:
        """Schedule a reminder (placeholder for future integration)"""
        return {
            "success": True,
            "message": f"Reminder scheduled for {customer_name}: {reminder_text}",
            "scheduled_at": datetime.now().isoformat()
        }

class AtlasAgent:
    """Main Atlas AI Agent"""
    
    def __init__(self, db_connection):
        self.db_conn = db_connection
        self.tools = AgentTools(db_connection)
        self.context = ConversationContext()
        self.openai_client = None
        self.knowledge = {}
        
        # Initialize OpenAI if API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load knowledge base from CSV (same dataset used by the Flask app)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, "data", "dataset.csv")
            if os.path.exists(csv_path):
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        prompt = (row.get("prompt", "") or "").strip().strip('"')
                        response = (row.get("response", "") or "").strip().strip('"')
                        norm = self._normalize(prompt)
                        if norm:
                            self.knowledge[norm] = response
        except Exception as e:
            # Don't crash if KB is unavailable; just proceed without it
            print(f"Knowledge base load failed: {e}")

    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        return (text.lower().strip()
                .translate(str.maketrans('', '', string.punctuation)))

    def _kb_lookup(self, user_input: str) -> str | None:
        """Return a KB answer if available (exact or close match)."""
        if not self.knowledge:
            return None
        norm = self._normalize(user_input)
        if not norm:
            return None
        # Exact
        if norm in self.knowledge:
            return self.knowledge[norm]
        # Fuzzy close match
        keys = list(self.knowledge.keys())
        matches = get_close_matches(norm, keys, n=1, cutoff=0.80)
        if matches:
            return self.knowledge[matches[0]]
        # Substring heuristic (short prompts)
        if len(norm) >= 6:
            for k in keys:
                if norm in k or k in norm:
                    return self.knowledge[k]
        return None
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user intent using pattern matching (fallback) or LLM"""
        
        if self.openai_client:
            return self._analyze_intent_with_llm(user_input)
        else:
            return self._analyze_intent_with_patterns(user_input)
    
    def _analyze_intent_with_patterns(self, user_input: str) -> Dict[str, Any]:
        user_input_lower = user_input.lower().strip()
        # Help/other commands patterns
        help_patterns = [
            r"^help$",
            r"^other commands$",
            r"^what else can you do$",
            r"^what can you do$",
            r"^commands$",
            r"^options$"
        ]
        for pattern in help_patterns:
            if re.match(pattern, user_input_lower):
                return {
                    "type": "greet",
                    "original": user_input
                }
        user_input_lower = user_input.lower().strip()
        # Delete customer patterns
        delete_patterns = [
            r"delete\s+([a-zA-Z][a-zA-Z'\- ]+)",
            r"remove\s+([a-zA-Z][a-zA-Z'\- ]+)"
        ]
        for pattern in delete_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                return {
                    "type": "delete_customer",
                    "name": match.group(1).strip(),
                    "original": user_input
                }

        # Edit/replace customer patterns (support individual fields)
        edit_patterns = [
            r"edit\s+([a-zA-Z][a-zA-Z'\- ]+)\s+(first name|last name|email|phone)\s+to\s+(.+)",
            r"update\s+([a-zA-Z][a-zA-Z'\- ]+)\s+(first name|last name|email|phone)\s+to\s+(.+)",
            r"edit\s+([a-zA-Z][a-zA-Z'\- ]+)\s+to\s+(.+)",
            r"replace\s+([a-zA-Z][a-zA-Z'\- ]+)\s+with\s+(.+)"
        ]
        for pattern in edit_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                if match.lastindex == 3:
                    # Individual field update
                    return {
                        "type": "edit_customer",
                        "name": match.group(1).strip(),
                        "field": match.group(2).strip(),
                        "value": match.group(3).strip(),
                        "original": user_input
                    }
                else:
                    # Full replace
                    return {
                        "type": "edit_customer",
                        "name": match.group(1).strip(),
                        "new_data": match.group(2).strip(),
                        "original": user_input
                    }
        """Pattern-based intent analysis (fallback)"""
        user_input_lower = user_input.lower().strip()
        
        # Greeting detection to provide a friendly welcome on startup/hello
        greet_keywords = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
        ]
        if (
            user_input_lower in greet_keywords
            or any(user_input_lower.startswith(k + " ") for k in greet_keywords)
        ):
            return {
                "type": "greet",
                "original": user_input
            }
        
        # Multi-step detection
        if "then" in user_input_lower or "and then" in user_input_lower:
            steps = re.split(r'\b(?:then|and then)\b', user_input_lower)
            return {
                "type": "multi_step",
                "steps": [step.strip() for step in steps],
                "original": user_input
            }
        
        # Knowledge base fallback early to avoid misclassifying domain questions as customer search
        kb_answer = self._kb_lookup(user_input)
        if kb_answer:
            return {
                "type": "kb",
                "answer": kb_answer,
                "original": user_input
            }
        
        # Search patterns
        search_patterns = [
            r"(?:find|search|get|show|tell me about)\s+(.+)",
            r"(?:contact|info|information)\s+(?:for|of)\s+(.+)",
            r"(.+?)(?:'s|s)?\s+(?:contact|email|phone|info|information)",
            r"who is (.+)",
            r"search (.+)"
        ]

        input_for_match = user_input.strip()
        for pattern in search_patterns:
            print(f"[DEBUG] Testing pattern: {pattern} on input: '{input_for_match}'")
            match = re.search(pattern, input_for_match, re.IGNORECASE)
            if match:
                print(f"[DEBUG] Pattern matched. Entity: '{match.group(1).strip()}'")
                return {
                    "type": "search_customer",
                    "entity": match.group(1).strip(),
                    "original": user_input
                }
        
        # Add customer patterns (non-greedy and anchored around the word 'customer')
        # Works with: "add new customer john doe [email] [phone]", "create customer jane smith"
        add_patterns = [
            r"(?:add|create|new)\s+(?:a\s+)?(?:new\s+)?customer\s+([a-z][a-z'\-]+)\s+([a-z][a-z'\-]+)(?:\s+([^\s@]+@[^\s@]+))?(?:\s+(\+?\d[\d\s\-]+))?$",
            r"customer\s+([a-z][a-z'\-]+)\s+([a-z][a-z'\-]+)(?:\s+([^\s@]+@[^\s@]+))?(?:\s+(\+?\d[\d\s\-]+))?$"
        ]
        
        for pattern in add_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                return {
                    "type": "add_customer",
                    "first_name": match.group(1),
                    "last_name": match.group(2),
                    "email": match.group(3) if match.group(3) else "",
                    "phone": match.group(4) if match.group(4) else "",
                    "original": user_input
                }
        
        # Stats patterns (placed before list to avoid misclassification)
        # Supports: "how many customers", "total customers", "are there 11 customers", "do we have 5 customers"
        stats_patterns = [
            r"(?:how many|number of|total)\s+customers",
            r"count\s+customers",
            r"are there\s+(\d+)\s+customers",
            r"do we have\s+(\d+)\s+customers"
        ]
        for pattern in stats_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                expected = None
                if match.lastindex:
                    try:
                        expected = int(match.group(1))
                    except Exception:
                        expected = None
                return {
                    "type": "get_stats",
                    "parameters": {"expected": expected} if expected is not None else {},
                    "original": user_input
                }

        # List patterns (avoid count questions)
        if (
            any(word in user_input_lower for word in ["list", "show all", "all customers"]) or
            ("customers" in user_input_lower and not any(kw in user_input_lower for kw in ["how many", "number of", "total", "count", "are there", "do we have"]))
        ):
            return {
                "type": "list_customers",
                "original": user_input
            }
        
        # (KB already checked earlier)
        
        return {
            "type": "unknown",
            "original": user_input
        }
    
    def _analyze_intent_with_llm(self, user_input: str) -> Dict[str, Any]:
        """LLM-based intent analysis"""
        try:
            system_prompt = """You are Atlas, an AI agent that helps manage customer information. 
            
Available actions:
- search_customer: Find customer by name
- add_customer: Add new customer (needs first_name, last_name, email, phone)
- list_customers: Show all customers
- get_stats: Get database statistics
- multi_step: Handle multi-step requests

Analyze the user's intent and respond with JSON in this format:
{
    "type": "action_name",
    "parameters": {"param": "value"},
    "confidence": 0.9,
    "explanation": "Brief explanation"
}

For multi-step requests, use:
{
    "type": "multi_step", 
    "steps": [{"type": "action1", "parameters": {}}, {"type": "action2", "parameters": {}}],
    "confidence": 0.9
}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            result["original"] = user_input
            return result
            
        except Exception as e:
            # Fallback to pattern matching
            print(f"LLM analysis failed: {e}")
            return self._analyze_intent_with_patterns(user_input)
    
    def execute_action(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        action_type = intent.get("type")
        if action_type == "delete_customer":
            name = intent.get("name", "")
            # Search for customer by name
            search_result = self.tools.search_customer(name)
            customers = search_result.get("results", [])
            if not customers:
                return {"success": False, "message": f"No customer found with name '{name}'"}
            # Delete all matches (could be multiple)
            deleted = 0
            for customer in customers:
                del_result = self.tools.delete_customer(customer["id"])
                if del_result.get("success"):
                    deleted += 1
            if deleted:
                return {"success": True, "message": f"Deleted {deleted} customer(s) named '{name}'"}
            else:
                return {"success": False, "message": f"Failed to delete customer(s) named '{name}'"}

        if action_type == "edit_customer":
            name = intent.get("name", "")
            # Individual field update
            if "field" in intent and "value" in intent:
                field_map = {
                    "first name": "first_name",
                    "last name": "last_name",
                    "email": "email",
                    "phone": "phone"
                }
                field = field_map.get(intent["field"].lower())
                value = intent["value"]
                if not field:
                    return {"success": False, "message": f"Unknown field '{intent['field']}'"}
                # Search for customer by name
                search_result = self.tools.search_customer(name)
                customers = search_result.get("results", [])
                if not customers:
                    return {"success": False, "message": f"No customer found with name '{name}'"}
                updated = 0
                for customer in customers:
                    upd_result = self.tools.update_customer(
                        customer["id"],
                        **{field: value}
                    )
                    if upd_result.get("success"):
                        updated += 1
                if updated:
                    return {"success": True, "message": f"Updated {updated} customer(s) named '{name}'"}
                else:
                    return {"success": False, "message": f"Failed to update customer(s) named '{name}'"}
            # Full replace (all fields)
            new_data = intent.get("new_data", "")
            parts = new_data.split()
            if len(parts) < 4:
                return {"success": False, "message": "Please provide new data as: first_name last_name email phone"}
            first_name, last_name, email, phone = parts[0], parts[1], parts[2], " ".join(parts[3:])
            # Capitalize first letter of each name part
            first_name = first_name.capitalize()
            last_name = last_name.capitalize()
            # Search for customer by name
            search_result = self.tools.search_customer(name)
            customers = search_result.get("results", [])
            if not customers:
                return {"success": False, "message": f"No customer found with name '{name}'"}
            updated = 0
            for customer in customers:
                upd_result = self.tools.update_customer(
                    customer["id"],
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone
                )
                if upd_result.get("success"):
                    updated += 1
            if updated:
                return {"success": True, "message": f"Updated {updated} customer(s) named '{name}'"}
            else:
                return {"success": False, "message": f"Failed to update customer(s) named '{name}'"}
        """Execute the determined action"""
        action_type = intent.get("type")

        if action_type == "greet":
            import random
            greetings = [
                "Hello! I'm Atlas, your assistant for customer management. How can I help you today?",
                "Welcome to Atlas! I can help you find, add, or list customers, and check database stats.",
                "Hi there! Atlas here. Need to search for a customer or add someone new? Just ask!",
                "Hey! I'm Atlas. Ready to help you manage your customer database. What would you like to do?",
                "Greetings! Atlas at your service. Try asking me to find a customer, add one, or list all customers."
            ]
            example_commands = (
                "\n\nYou can try simple commands first like:\n"
                "- Search: 'Find Sandra' or, Delete: 'delete Jane'\n"
                "- Add a customer: 'Add customer John Doe john@example.com 321-555-1234'\n"

                "\nThen try more complex features like:\n"
                "- Update phone or, update email: 'edit David email to david.brown23@example.com'\n"
                "- Replace all info: 'edit David to Dave Brown dave@email.com 232-555-1234'\n"
                "- Upload a CSV: Upload a file with columns first_name, last_name, email, phone\n"
            )
            return {
                "success": True,
                "type": "greet",
                "message": f"{random.choice(greetings)}{example_commands}"
            }

        if action_type == "kb":
            return {
                "success": True,
                "type": "kb",
                "message": intent.get("answer", "")
            }

        if action_type == "search_customer":
            entity = intent.get("entity") or intent.get("parameters", {}).get("name", "")
            return self.tools.search_customer(entity)
        
        elif action_type == "add_customer":
            # Get parameters from pattern matching
            first_name = (intent.get("first_name", "") or "").strip()
            last_name = (intent.get("last_name", "") or "").strip()
            # Capitalize first letter of each name part
            first_name = first_name.capitalize()
            last_name = last_name.capitalize()
            email = intent.get("email", "")
            phone = intent.get("phone", "")
            # Combine first and last name for the method call
            full_name = f"{first_name} {last_name}".strip()
            return self.tools.add_customer(full_name, email, phone)
        
        elif action_type == "list_customers":
            limit = intent.get("parameters", {}).get("limit", 10)
            return self.tools.get_customers(limit)
        
        elif action_type == "get_stats":
            return self.tools.get_customer_stats()
        
        elif action_type == "multi_step":
            return self._execute_multi_step(intent.get("steps", []))
        
        else:
            return {
                "success": False,
                "message": "I'm not sure how to help with that. I can help you search for customers, add new customers, list all customers, or get database statistics.",
                "suggestions": [
                    "Search for a customer: 'Find Sandra'",
                    "Add customer: 'Add customer Joe Smith josmith@email.com 321-555-1234'",
                    "List customers: 'Show all customers'",
                    "Get stats: 'How many customers do we have?'"
                ]
            }
    
    def _execute_multi_step(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple steps in sequence"""
        results = []
        for step in steps:
            result = self.execute_action(step)
            results.append(result)
            
            # Stop if any step fails
            if not result.get("success", True):
                break
        
        return {
            "success": all(r.get("success", True) for r in results),
            "steps": results,
            "type": "multi_step"
        }
    
    def format_response(self, result: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Format the response for display"""
        if not result.get("success", True):
            error_msg = result.get("message") or result.get("error") or "An error occurred"
            suggestions = result.get("suggestions") or []
            if suggestions:
                tips = "\n" + "\n".join([f"- {s}" for s in suggestions])
            else:
                tips = ""
            return f"Sorry, I encountered an error: {error_msg}{tips}"
        
        result_type = result.get("type") or intent.get("type")
        
        if result_type == "greet":
            return result.get("message", "Hello! How can I help you today?")
        if result_type == "kb":
            return result.get("message", "")

        if result_type == "search_customer":
            customers = result.get("results", [])
            if not customers:
                return f"I couldn't find any customers matching your search. Would you like me to add a new customer? If so, please provide the details in the following format: 'Add customer John Doe john@email.com 212-555-1234'"
            
            if len(customers) == 1:
                customer = customers[0]
                return (
                    f"Name: {customer['first_name']} {customer['last_name']}\n"
                    f"Email: {customer['email']}\n"
                    f"Phone: {customer['phone']}"
                )
            
            response = f"I found {len(customers)} customers:\n"
            for customer in customers[:5]:  # Limit to 5 results
                response += f"• {customer['first_name']} {customer['last_name']} - {customer['email']}\n"
            
            if len(customers) > 5:
                response += f"... and {len(customers) - 5} more results."
            
            return response
        
        elif result_type == "add_customer":
            return result.get("message", "Customer added successfully!")
        
        elif result_type == "list_customers":
            customers = result.get("results") if result.get("results") is not None else result.get("customers", [])
            if not customers:
                return "No customers found in the database."
            
            response = f"Here are the latest {len(customers)} customers:\n"
            for customer in customers:
                first = (customer.get('first_name') or '').strip()
                last = (customer.get('last_name') or '').strip()
                email = (customer.get('email') or '').strip()
                name = (first + ' ' + last).strip()
                if email:
                    response += f"• {name} - {email}\n"
                else:
                    response += f"• {name}\n"
            
            return response
        
        elif result_type == "get_stats":
            total = result.get("total_customers", 0)
            expected = intent.get("parameters", {}).get("expected") if intent.get("parameters") else None
            if expected is not None:
                if expected == total:
                    return f"Yes, there are {total} customers."
                else:
                    return f"No, there are {total} customers (not {expected})."
            # Default stats response
            return f"We currently have {total} customers."
        
        elif result_type == "multi_step":
            responses = []
            for i, step_result in enumerate(result.get("steps", []), 1):
                step_response = self.format_response(step_result, {"type": step_result.get("type", "unknown")})
                responses.append(f"Step {i}: {step_response}")
            
            return "\n\n".join(responses)
        
        return result.get("message", "Action completed successfully!")
    
    def process_message(self, user_input: str) -> str:
        """Main entry point for processing user messages"""
        # Add to conversation history
        self.context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "type": "user"
        })
        
        # Analyze intent
        intent = self.analyze_intent(user_input)
        
        # Execute action
        result = self.execute_action(intent)
        
        # Format response
        response = self.format_response(result, intent)
        
        # Add response to conversation history
        self.context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "bot": response,
            "type": "assistant",
            "intent": intent,
            "result": result
        })
        
        # Keep conversation history manageable
        if len(self.context.conversation_history) > 40:  # 20 exchanges
            self.context.conversation_history = self.context.conversation_history[-40:]
        
        return response