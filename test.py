"""
Verification script to check if chat history is properly passed to LLM
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_chat_history():
    """Test if chat history is being used by asking same question twice"""
    
    session_id = f"test_session_{int(time.time())}"
    
    print("="*70)
    print("CHAT HISTORY VERIFICATION TEST")
    print("="*70)
    
    # First query
    print("\n📤 QUERY 1: List all U&A reports")
    response1 = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "List all U&A reports",
            "session_id": session_id
        }
    )
    
    result1 = response1.json()
    print(f"✅ Response 1 received ({len(result1.get('response', ''))} chars)")
    print(f"📊 Sources: {result1.get('metadata', {}).get('sources', 0)}")
    
    time.sleep(2)
    
    # Second query - SAME question
    print("\n📤 QUERY 2: List all U&A reports (SAME QUESTION)")
    response2 = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "List all U&A reports",
            "session_id": session_id
        }
    )
    
    result2 = response2.json()
    print(f"✅ Response 2 received ({len(result2.get('response', ''))} chars)")
    print(f"📊 Sources: {result2.get('metadata', {}).get('sources', 0)}")
    
    # Check history
    print("\n📜 CHECKING CONVERSATION HISTORY...")
    history = requests.get(f"{BASE_URL}/history/{session_id}")
    history_data = history.json()
    
    print(f"Total messages in history: {len(history_data.get('messages', []))}")
    print("\nMessage types:")
    for i, msg in enumerate(history_data.get('messages', [])):
        print(f"  {i+1}. {msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}...")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if result2.get('metadata', {}).get('sources', 0) > 0:
        print("❌ FAIL: Second query retrieved new documents")
        print("   Expected: LLM should reference previous response without retrieval")
        print("   Actual: LLM made new search (sources > 0)")
        print("\n💡 This means:")
        print("   - History is stored BUT NOT being used by LLM")
        print("   - LLM is not seeing previous messages")
        print("   - Check if MessagesPlaceholder is working")
    else:
        print("✅ PASS: Second query used cached/previous response")
        print("   LLM referenced history without new retrieval")
    
    return result1, result2, history_data


def test_follow_up_context():
    """Test if follow-up questions use context"""
    
    session_id = f"test_session_{int(time.time())}"
    
    print("\n" + "="*70)
    print("FOLLOW-UP CONTEXT TEST")
    print("="*70)
    
    # First query
    print("\n📤 QUERY 1: What is Godrej No.1 market share?")
    response1 = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "What is Godrej No.1 market share?",
            "session_id": session_id
        }
    )
    
    result1 = response1.json()
    print(f"✅ Response 1 received")
    print(f"📊 Sources: {result1.get('metadata', {}).get('sources', 0)}")
    
    time.sleep(2)
    
    # Follow-up query (ambiguous - requires context)
    print("\n📤 QUERY 2: What about Lux? (ambiguous follow-up)")
    response2 = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "What about Lux?",
            "session_id": session_id
        }
    )
    
    result2 = response2.json()
    print(f"✅ Response 2 received")
    print(f"📊 Sources: {result2.get('metadata', {}).get('sources', 0)}")
    
    # Check if response mentions "market share" (inherited context)
    response_text = result2.get('response', '').lower()
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if 'market share' in response_text or 'share' in response_text:
        print("✅ PASS: LLM understood context from previous query")
        print("   'What about Lux?' was interpreted as 'What is Lux market share?'")
        print("   This means chat history IS being used!")
    else:
        print("❌ FAIL: LLM did not use context")
        print("   Response doesn't mention market share")
        print("   LLM treated 'What about Lux?' as isolated query")
    
    return result1, result2


if __name__ == "__main__":
    print("\n🔍 Starting Chat History Verification Tests...")
    print("Make sure your Flask server is running on http://localhost:5001\n")
    
    try:
        # Test 1: Same question twice
        test_chat_history()
        
        time.sleep(3)
        
        # Test 2: Follow-up context
        test_follow_up_context()
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to Flask server")
        print("Make sure the server is running: python app.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# #!/usr/bin/env python3
# """
# Database Setup Script
# Creates the qt328pp database and initializes tables
# """

# import os
# import sys
# from dotenv import load_dotenv

# # Try to import psycopg
# try:
#     import psycopg
#     from psycopg import sql
# except ImportError:
#     print("❌ psycopg not installed!")
#     print("Run: pip install 'psycopg[binary,pool]'")
#     sys.exit(1)

# load_dotenv()

# # Database configuration
# POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
# POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
# POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
# POSTGRES_DB = os.getenv("POSTGRES_DB", "qt328pp")

# print("🔧 PostgreSQL Database Setup")
# print("=" * 50)
# print(f"Host: {POSTGRES_HOST}")
# print(f"Port: {POSTGRES_PORT}")
# print(f"User: {POSTGRES_USER}")
# print(f"Database: {POSTGRES_DB}")
# print("=" * 50)

# # Step 1: Connect to PostgreSQL server (default 'postgres' database)
# print("\n📡 Connecting to PostgreSQL server...")
# try:
#     # Connect to default postgres database first
#     conn = psycopg.connect(
#         host=POSTGRES_HOST,
#         port=POSTGRES_PORT,
#         user=POSTGRES_USER,
#         password=POSTGRES_PASSWORD,
#         dbname="postgres",  # Connect to default database first
#         autocommit=True  # Required for CREATE DATABASE
#     )
#     print("✅ Connected to PostgreSQL server")
# except Exception as e:
#     print(f"❌ Failed to connect to PostgreSQL server: {e}")
#     print("\n💡 Troubleshooting:")
#     print("1. Is PostgreSQL running?")
#     print("   - Linux: sudo systemctl status postgresql")
#     print("   - macOS: brew services list")
#     print("   - Windows: Check Services")
#     print("\n2. Check your credentials in .env file:")
#     print(f"   POSTGRES_USER={POSTGRES_USER}")
#     print(f"   POSTGRES_PASSWORD=***")
#     print(f"   POSTGRES_HOST={POSTGRES_HOST}")
#     print(f"   POSTGRES_PORT={POSTGRES_PORT}")
#     sys.exit(1)

# # Step 2: Check if database exists
# print(f"\n🔍 Checking if database '{POSTGRES_DB}' exists...")
# try:
#     with conn.cursor() as cur:
#         cur.execute(
#             "SELECT 1 FROM pg_database WHERE datname = %s",
#             (POSTGRES_DB,)
#         )
#         exists = cur.fetchone()
        
#         if exists:
#             print(f"✅ Database '{POSTGRES_DB}' already exists")
#         else:
#             # Step 3: Create database
#             print(f"🏗️  Creating database '{POSTGRES_DB}'...")
#             cur.execute(
#                 sql.SQL("CREATE DATABASE {}").format(
#                     sql.Identifier(POSTGRES_DB)
#                 )
#             )
#             print(f"✅ Database '{POSTGRES_DB}' created successfully!")
# except Exception as e:
#     print(f"❌ Error checking/creating database: {e}")
#     conn.close()
#     sys.exit(1)

# conn.close()

# # Step 4: Connect to the new database and initialize tables
# print(f"\n📊 Connecting to '{POSTGRES_DB}' to initialize tables...")
# try:
#     conn = psycopg.connect(
#         host=POSTGRES_HOST,
#         port=POSTGRES_PORT,
#         user=POSTGRES_USER,
#         password=POSTGRES_PASSWORD,
#         dbname=POSTGRES_DB
#     )
#     print(f"✅ Connected to '{POSTGRES_DB}'")
# except Exception as e:
#     print(f"❌ Failed to connect to '{POSTGRES_DB}': {e}")
#     sys.exit(1)

# # Step 5: Initialize LangGraph checkpoint tables
# print("\n🏗️  Initializing LangGraph checkpoint tables...")
# try:
#     from langgraph.checkpoint.postgres import PostgresSaver
#     from psycopg_pool import ConnectionPool
    
#     # Create connection pool
#     pool = ConnectionPool(
#         conninfo=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
#         kwargs={"autocommit": True, "prepare_threshold": 0},
#         max_size=20
#     )
    
#     # Setup checkpointer tables
#     checkpointer = PostgresSaver(pool)
#     checkpointer.setup()
    
#     print("✅ LangGraph checkpoint tables created")
# except Exception as e:
#     print(f"⚠️  Warning: Could not initialize checkpoint tables: {e}")

# # Step 6: Initialize PostgresStore tables
# print("\n🏗️  Initializing PostgresStore tables...")
# try:
#     from langgraph.store.postgres import PostgresStore
#     from psycopg_pool import ConnectionPool
    
#     # Create connection pool for store
#     store_pool = ConnectionPool(
#         conninfo=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
#         kwargs={"autocommit": True},
#         max_size=10
#     )
    
#     # Setup store tables
#     store = PostgresStore(store_pool)
#     store.setup()
    
#     print("✅ PostgresStore tables created")
# except Exception as e:
#     print(f"⚠️  Warning: Could not initialize store tables: {e}")

# # Step 7: Verify tables were created
# print("\n📋 Verifying tables...")
# try:
#     with conn.cursor() as cur:
#         cur.execute("""
#             SELECT table_name 
#             FROM information_schema.tables 
#             WHERE table_schema = 'public'
#             ORDER BY table_name;
#         """)
#         tables = cur.fetchall()
        
#         if tables:
#             print(f"✅ Found {len(tables)} tables:")
#             for table in tables:
#                 print(f"   - {table[0]}")
#         else:
#             print("⚠️  No tables found")
# except Exception as e:
#     print(f"❌ Error listing tables: {e}")

# conn.close()

# print("\n" + "=" * 50)
# print("✅ Database setup complete!")
# print("=" * 50)
# print(f"\n📝 Connection details for VSCode:")
# print(f"   Host: {POSTGRES_HOST}")
# print(f"   Port: {POSTGRES_PORT}")
# print(f"   Username: {POSTGRES_USER}")
# print(f"   Password: {POSTGRES_PASSWORD}")
# print(f"   Database: {POSTGRES_DB}")
# print("\n🚀 You can now connect from VSCode!")