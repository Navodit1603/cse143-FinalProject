from flask import Flask, render_template, request
from madlib_functions import generate_madlib  # Import the function from madlib_functions.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    replacements = {}
    dynamic_pos_names = {}

    if request.method == 'POST':
        # Get user input for replacements
        for key in request.form:
            replacements[key] = request.form.get(key, "")

        # Generate Madlib using the imported function
        dynamic_pos_names, output = generate_madlib(replacements)

    return render_template('index.html', output=output, pos_names=dynamic_pos_names)

if __name__ == '__main__':
    app.run(debug=True)
