from flask import Flask, render_template, request
from madlib_functions import get_pos_names, generate_madlib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    replacements = {}
    dynamic_pos_names = {}
    dynamic_pos_names = get_pos_names()

    if request.method == 'POST':
        for key in request.form:
            replacements[key] = request.form.get(key, "")
        output = generate_madlib(replacements)
    print(output)
    return render_template('index.html', pos_names=dynamic_pos_names, output=output)

if __name__ == '__main__':
    app.run(debug=True)
