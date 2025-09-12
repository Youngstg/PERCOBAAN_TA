from flask import Flask, render_template


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


app = create_app()


if __name__ == "__main__":
    # Run the app for local development
    # Access via: http://127.0.0.1:5000/
    app.run(debug=True)

