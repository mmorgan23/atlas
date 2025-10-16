from flask import Flask, render_template, request, redirect, url_for, flash, get_flashed_messages
from livereload import Server
import csv
import string
import psycopg2
import psycopg2.extras
import io
import re
import os
from atlas_agent import AtlasAgent
from dotenv import load_dotenv



load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'atlas-secret-key')

# Connect to PostgreSQL
db_conn = psycopg2.connect(
    host=os.getenv('PG_HOST', 'localhost'),
    user=os.getenv('PG_USER', 'atlas_user'),
    password=os.getenv('PG_PASSWORD', ''),
    port=int(os.getenv('PG_PORT', 5432)),
    dbname=os.getenv('PG_DB', 'atlas_customers')
)

# Initialize Atlas Agent
atlas_agent = AtlasAgent(db_conn)

print("✅ Atlas AI Agent initialized successfully!")
print("🤖 Level 1 Agent Features:")
print("   • Natural Language Understanding")  
print("   • Multi-step Conversations")
print("   • Context Awareness") 
print("   • Enhanced Error Handling")

def normalize(text):
    return text.lower().strip().translate(str.maketrans('', '', string.punctuation))

def extract_name_from_query(query):
    """Extract name from contact information queries"""
    # Convert to lowercase for processing but keep original case for names
    lower_query = query.lower().strip()
    
    # Common patterns for contact queries
    patterns = [
        r"(?:what is|get|find|show|tell me)\s+(.+?)(?:'s|s)?\s+(?:contact|email|phone|info|information)",
        r"(?:contact|email|phone|info|information)\s+(?:for|of)\s+(.+)",
        r"(.+?)(?:'s|s)?\s+(?:contact|email|phone|info|information)",
        r"find\s+(.+?)\s+contact",
        r"(.+?)\s+contact\s+(?:info|information)",
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, lower_query)
        if match:
            name = match.group(1).strip()
            # Clean up common words
            name = re.sub(r'\b(?:the|a|an|for|of|is|what|get|find|show|tell|me)\b', '', name).strip()
            # Remove extra whitespace
            name = re.sub(r'\s+', ' ', name)
            return name.strip()
    
    return None

def is_likely_person_name(text):
    """Check if the input is likely a person's name (first name or first + last name)"""
    # Clean the input
    clean_text = text.strip()
    
    # Skip if it's too long (more than 4 words is unlikely to be just a name)
    words = clean_text.split()
    if len(words) > 4 or len(words) == 0:
        return False
    
    # Skip if it contains common non-name words or phrases
    non_name_indicators = [
        'hello', 'hi', 'hey', 'thanks', 'thank you', 'please', 'help',
        'what', 'how', 'when', 'where', 'why', 'who', 'can', 'could',
        'would', 'should', 'do', 'does', 'did', 'is', 'are', 'am', 'was', 'were',
        'the', 'a', 'an', 'and', 'or', 'but', 'so', 'if', 'then', 'that', 'this',
        'contact', 'email', 'phone', 'information', 'info', 'get', 'find', 'show', 'tell'
    ]
    
    text_lower = clean_text.lower()
    
    # Check for exact matches with non-name indicators (not partial matches)
    for indicator in non_name_indicators:
        if indicator == text_lower or f" {indicator} " in f" {text_lower} " or text_lower.startswith(f"{indicator} ") or text_lower.endswith(f" {indicator}"):
            return False
    
    # Check if words look like names (should be alphabetic characters only)
    for word in words:
        clean_word = word.replace("'", "").replace("-", "").replace(".", "")
        if not clean_word.isalpha() or len(clean_word) < 2:
            return False
    
    # Additional check: if it's a single word that's very common, it might not be a name
    # But we'll be more permissive here since names can be common words
    if len(words) == 1:
        # Very short words are unlikely to be names
        if len(clean_text) < 2:
            return False
    
    return True

def search_person_by_name(name_input):
    """Search for a person in the database by name"""
    name_parts = name_input.strip().split()
    
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        if len(name_parts) == 1:
            # Only first name provided
            first_name = name_parts[0].title()
            cursor.execute(
                "SELECT * FROM customers WHERE LOWER(first_name) = LOWER(%s)",
                (first_name,)
            )
            matches = cursor.fetchall()
            cursor.close()
            
            if len(matches) == 0:
                return f"Sorry, I couldn't find anyone named {first_name}."
            elif len(matches) == 1:
                customer = matches[0]
                return (
                    f"Name: {customer['first_name']} {customer['last_name']}\n"
                    f"Email: {customer['email']}\n"
                    f"Phone: {customer['phone']}"
                )
            else:
                # Multiple matches
                names = [f"{c['first_name']} {c['last_name']}" for c in matches]
                return f"I found multiple people named {first_name}: {', '.join(names)}. Please be more specific."
        else:
            # Full name provided
            first_name = name_parts[0].title()
            last_name = " ".join(name_parts[1:]).title()
            cursor.execute(
                "SELECT * FROM customers WHERE LOWER(first_name) = LOWER(%s) AND LOWER(last_name) = LOWER(%s)",
                (first_name, last_name)
            )
            customer = cursor.fetchone()
            cursor.close()
            
            if customer:
                return (
                    f"Name: {customer['first_name']} {customer['last_name']}\n"
                    f"Email: {customer['email']}\n"
                    f"Phone: {customer['phone']}"
                )
            else:
                return f"Sorry, I couldn't find contact info for {first_name} {last_name}."
    except Exception as e:
        return f"Database error: {e}"

# Load knowledge base from CSV
knowledge = {}
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data", "dataset.csv")
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt = normalize(row.get("prompt", "").strip('"').strip())
        response = row.get("response", "").strip('"').strip()
        if prompt:
            knowledge[prompt] = response

# Global variable for chat history - cleared on server restart
history = []
print("✅ Server started: Chat history cleared")

@app.route("/add_customer", methods=["POST"])
def add_customer():
    """Route to add a single customer"""
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    
    if not all([first_name, last_name, email, phone]):
        flash("Please fill in all fields.")
        return redirect(url_for('chat'))

    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "INSERT INTO customers (first_name, last_name, email, phone) VALUES (%s, %s, %s, %s)",
            (first_name, last_name, email, phone)
        )
        db_conn.commit()
        cursor.close()
        flash(f"Customer {first_name} {last_name} added successfully!")
    except Exception as e:
        flash(f"Error adding customer: {e}")
    return redirect(url_for('chat'))
    # return render_template("chat.html", message=message, user_input="", bot_response="", history=history, customers=customers)

@app.route("/upload_customers", methods=["POST"])
def upload_customers():
    """Route to upload customers from CSV"""
    if 'csv_file' not in request.files:
        try:
            cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM customers")
            customers = cursor.fetchall()
            cursor.close()
        except Exception as e:
            customers = [{"error": str(e)}]
        return render_template("chat.html", message="No file selected.", user_input="", bot_response="", history=history, customers=customers)
    
    file = request.files['csv_file']
    if file.filename == '':
        try:
            cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM customers")
            customers = cursor.fetchall()
            cursor.close()
        except Exception as e:
            customers = [{"error": str(e)}]
        return render_template("chat.html", message="No file selected.", user_input="", bot_response="", history=history, customers=customers)
    
    if not file.filename.lower().endswith('.csv'):
        try:
            cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM customers")
            customers = cursor.fetchall()
            cursor.close()
        except Exception as e:
            customers = [{"error": str(e)}]
        return render_template("chat.html", message="Please upload a CSV file.", user_input="", bot_response="", history=history, customers=customers)
    
    try:
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        
        added_count = 0
        cursor = db_conn.cursor()
        
        for row in csv_input:
            first_name = row.get("first_name", "").strip()
            last_name = row.get("last_name", "").strip()
            email = row.get("email", "").strip()
            phone = row.get("phone", "").strip()
            
            if all([first_name, last_name, email, phone]):
                try:
                    cursor.execute(
                        "INSERT INTO customers (first_name, last_name, email, phone) VALUES (%s, %s, %s, %s)",
                        (first_name, last_name, email, phone)
                    )
                    added_count += 1
                except Exception as e:
                    print(f"Error adding row: {e}")
        
        db_conn.commit()
        cursor.close()
        message = f"Successfully added {added_count} customers from CSV."
        
    except Exception as e:
        message = f"Error processing CSV: {e}"
    
    # Fetch customers to display
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM customers")
        customers = cursor.fetchall()
        cursor.close()
    except Exception as e:
        customers = [{"error": str(e)}]
    
    flash(message)
    return redirect(url_for('chat'))

@app.route("/clear", methods=["POST"])
def clear_chat():
    """Route to clear chat history"""
    global history
    history = []
    return render_template("chat.html", message="Chat cleared.", user_input="", bot_response="", history=[], customers=[])

@app.route("/livereload", methods=["GET"])
def livereload():
    """Handle livereload requests to avoid 404s"""
    return "", 204

@app.route("/", methods=["GET", "POST"])
def chat():
    global history
    
    if request.method == "POST":
        user_input = request.form.get("message", "")
        if user_input:
            # Use Atlas Agent to process the message
            bot_response = atlas_agent.process_message(user_input)
            
            # Add to Flask history for template compatibility
            history.append({"user": user_input, "bot": bot_response})
            
            # Limit history to last 20 exchanges
            if len(history) > 20:
                history = history[-20:]
        
        # POST-Redirect-GET: Redirect to prevent resubmission on refresh
        return redirect(url_for('chat'))
    
    # GET request - just display the chat
    user_input = ""
    bot_response = ""
    customers = []
    message = None
    flash_messages = get_flashed_messages()

    # Initialize with agent greeting if no history
    if not history:
        bot_response = atlas_agent.process_message("hello")  # Get agent's greeting
        history.append({"user": "", "bot": bot_response})

    # Fetch all customers from MySQL
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM customers")
        customers = cursor.fetchall()
        cursor.close()
        print("Fetched customers:", customers)  # Debugging statement
    except Exception as e:
        customers = [{"error": str(e)}]
        print("Fetched customers:", customers)  # Debugging statement added here

    return render_template("chat.html", user_input=user_input, bot_response=bot_response, history=history, customers=customers, message=message, flash_messages=flash_messages)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)