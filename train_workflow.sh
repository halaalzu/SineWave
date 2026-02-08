#!/bin/bash

# FlowState Training Workflow
# Complete pipeline from data collection to model training

set -e

echo "════════════════════════════════════════════════════════════"
echo "  FlowState Training Pipeline"
echo "════════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found"
    echo "Run: python3 -m venv .venv"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Menu
echo "Choose an option:"
echo ""
echo "  1. Generate synthetic training data (for testing)"
echo "  2. Start web app to collect real data"
echo "  3. Train ML models"
echo "  4. Setup PostgreSQL + pgvector"
echo "  5. Create embeddings for vector search"
echo "  6. Demo similarity search"
echo "  7. Complete pipeline (generate data + train + embeddings)"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "Generating synthetic training data..."
        read -p "Sessions per user (5-20 recommended): " sessions
        sessions=${sessions:-5}
        python populate_data.py $sessions
        ;;
    
    2)
        echo ""
        echo "Starting web app on http://localhost:5001"
        echo "Press CTRL+C to stop"
        echo ""
        python app_with_data.py
        ;;
    
    3)
        echo ""
        echo "Training ML models..."
        python train_models.py
        ;;
    
    4)
        echo ""
        echo "Setting up PostgreSQL with pgvector..."
        
        # Check if PostgreSQL is running
        if ! pgrep -x "postgres" > /dev/null; then
            echo "Starting PostgreSQL..."
            brew services start postgresql || true
            sleep 2
        fi
        
        # Create database if needed
        if ! psql -lqt | cut -d \| -f 1 | grep -qw flowstate; then
            echo "Creating database..."
            createdb flowstate
        fi
        
        # Enable pgvector and create tables
        python database_postgres.py
        
        # Ask about migration
        read -p "Migrate data from SQLite to PostgreSQL? [y/N]: " migrate
        if [[ $migrate == "y" || $migrate == "Y" ]]; then
            python -c "from database_postgres import migrate_sqlite_to_postgres; migrate_sqlite_to_postgres('flowstate.db')"
        fi
        ;;
    
    5)
        echo ""
        echo "Creating embeddings for vector search..."
        python pgvector_integration.py create
        ;;
    
    6)
        echo ""
        echo "Running similarity search demo..."
        python pgvector_integration.py demo
        ;;
    
    7)
        echo ""
        echo "═══════════════════════════════════════"
        echo "  Complete Training Pipeline"
        echo "═══════════════════════════════════════"
        echo ""
        
        # Step 1: Generate data
        echo "Step 1: Generating synthetic data..."
        python populate_data.py 10
        
        # Step 2: Train models
        echo ""
        echo "Step 2: Training ML models..."
        python train_models.py
        
        # Step 3: Optional PostgreSQL
        read -p "Setup PostgreSQL + pgvector? [y/N]: " setup_pg
        if [[ $setup_pg == "y" || $setup_pg == "Y" ]]; then
            if ! pgrep -x "postgres" > /dev/null; then
                brew services start postgresql || true
                sleep 2
            fi
            
            if ! psql -lqt | cut -d \| -f 1 | grep -qw flowstate; then
                createdb flowstate
            fi
            
            python database_postgres.py
            python -c "from database_postgres import migrate_sqlite_to_postgres; migrate_sqlite_to_postgres('flowstate.db')"
            
            echo ""
            echo "Step 3: Creating embeddings..."
            python pgvector_integration.py create
        fi
        
        echo ""
        echo "═══════════════════════════════════════"
        echo "  ✓ Pipeline Complete!"
        echo "═══════════════════════════════════════"
        echo ""
        echo "Your models are ready in the models/ directory"
        echo ""
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"
