#!/usr/bin/env python3
"""
Database Setup Script
Creates the qt328pp database and initializes tables
"""

import os
import sys
from dotenv import load_dotenv

# Try to import psycopg
try:
    import psycopg
    from psycopg import sql
except ImportError:
    print("❌ psycopg not installed!")
    print("Run: pip install 'psycopg[binary,pool]'")
    sys.exit(1)

load_dotenv()

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "qt328pp")

print("🔧 PostgreSQL Database Setup")
print("=" * 50)
print(f"Host: {POSTGRES_HOST}")
print(f"Port: {POSTGRES_PORT}")
print(f"User: {POSTGRES_USER}")
print(f"Database: {POSTGRES_DB}")
print("=" * 50)

# Step 1: Connect to PostgreSQL server (default 'postgres' database)
print("\n📡 Connecting to PostgreSQL server...")
try:
    # Connect to default postgres database first
    conn = psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname="postgres",  # Connect to default database first
        autocommit=True  # Required for CREATE DATABASE
    )
    print("✅ Connected to PostgreSQL server")
except Exception as e:
    print(f"❌ Failed to connect to PostgreSQL server: {e}")
    print("\n💡 Troubleshooting:")
    print("1. Is PostgreSQL running?")
    print("   - Linux: sudo systemctl status postgresql")
    print("   - macOS: brew services list")
    print("   - Windows: Check Services")
    print("\n2. Check your credentials in .env file:")
    print(f"   POSTGRES_USER={POSTGRES_USER}")
    print(f"   POSTGRES_PASSWORD=***")
    print(f"   POSTGRES_HOST={POSTGRES_HOST}")
    print(f"   POSTGRES_PORT={POSTGRES_PORT}")
    sys.exit(1)

# Step 2: Check if database exists
print(f"\n🔍 Checking if database '{POSTGRES_DB}' exists...")
try:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (POSTGRES_DB,)
        )
        exists = cur.fetchone()
        
        if exists:
            print(f"✅ Database '{POSTGRES_DB}' already exists")
        else:
            # Step 3: Create database
            print(f"🏗️  Creating database '{POSTGRES_DB}'...")
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(POSTGRES_DB)
                )
            )
            print(f"✅ Database '{POSTGRES_DB}' created successfully!")
except Exception as e:
    print(f"❌ Error checking/creating database: {e}")
    conn.close()
    sys.exit(1)

conn.close()

# Step 4: Connect to the new database and initialize tables
print(f"\n📊 Connecting to '{POSTGRES_DB}' to initialize tables...")
try:
    conn = psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB
    )
    print(f"✅ Connected to '{POSTGRES_DB}'")
except Exception as e:
    print(f"❌ Failed to connect to '{POSTGRES_DB}': {e}")
    sys.exit(1)

# Step 5: Initialize LangGraph checkpoint tables
print("\n🏗️  Initializing LangGraph checkpoint tables...")
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    
    # Create connection pool
    pool = ConnectionPool(
        conninfo=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        kwargs={"autocommit": True, "prepare_threshold": 0},
        max_size=20
    )
    
    # Setup checkpointer tables
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    
    print("✅ LangGraph checkpoint tables created")
except Exception as e:
    print(f"⚠️  Warning: Could not initialize checkpoint tables: {e}")

# Step 6: Initialize PostgresStore tables
print("\n🏗️  Initializing PostgresStore tables...")
try:
    from langgraph.store.postgres import PostgresStore
    from psycopg_pool import ConnectionPool
    
    # Create connection pool for store
    store_pool = ConnectionPool(
        conninfo=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        kwargs={"autocommit": True},
        max_size=10
    )
    
    # Setup store tables
    store = PostgresStore(store_pool)
    store.setup()
    
    print("✅ PostgresStore tables created")
except Exception as e:
    print(f"⚠️  Warning: Could not initialize store tables: {e}")

# Step 7: Verify tables were created
print("\n📋 Verifying tables...")
try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        
        if tables:
            print(f"✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("⚠️  No tables found")
except Exception as e:
    print(f"❌ Error listing tables: {e}")

conn.close()

print("\n" + "=" * 50)
print("✅ Database setup complete!")
print("=" * 50)
print(f"\n📝 Connection details for VSCode:")
print(f"   Host: {POSTGRES_HOST}")
print(f"   Port: {POSTGRES_PORT}")
print(f"   Username: {POSTGRES_USER}")
print(f"   Password: {POSTGRES_PASSWORD}")
print(f"   Database: {POSTGRES_DB}")
print("\n🚀 You can now connect from VSCode!")