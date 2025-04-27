from bigTp import app
import os

if __name__ == '__main__':
    app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app.run(debug=True)