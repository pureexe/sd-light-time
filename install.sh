# create virtual environment if not exist

# check if virtual environment exists in python/venv
# if not, create virtual environment and install dependencies
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Virtual environment exists."
fi