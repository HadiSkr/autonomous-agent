"""Chrome-GPT: An AutoGPT agent that interacts with Chrome"""
import click
from flask import Flask, request, jsonify
from chromegpt.main import run_chromegpt


# @click.command()
# @click.option("--task", "-t", help="The task to execute", required=True)
# @click.option(
#     "--agent",
#     "-a",
#     help="The agent type to use",
#     default="zero-shot",
#     type=click.Choice(["auto-gpt", "baby-agi", "zero-shot"], case_sensitive=False),
# )
# @click.option("--model", "-m", help="The model to use", default="gpt-3.5-turbo")
# @click.option("--headless", help="Run in headless mode", is_flag=True)
# @click.option("--verbose", "-v", help="Run in verbose mode", is_flag=True)
# @click.option(
#     "--human-in-loop",
#     help="Run in human-in-loop mode, only available when using auto-gpt agent",
#     is_flag=True,
# )

# def main(
#     task: str,
#     agent: str,
#     model: str = "gpt-3.5-turbo",
#     headless: bool = False,
#     verbose: bool = False,
#     human_in_loop: bool = False,
# ) -> str:
#     """Run ChromeGPT: An AutoGPT agent that interacts with Chrome"""
#     return run_chromegpt(
#         task=task,
#         model=model,
#         agent=agent,
#         headless=headless,
#         verbose=verbose,
#         continuous=not human_in_loop,
#     )


# if __name__ == "__main__":
#     main()
app = Flask(__name__)
# @app.route('/')
# def home():
#     return "Hello, World!"

@app.route('/run', methods=['POST'])
def run_query():
    data = request.get_json()
    user_query = data['query']
    print("Received query: ", user_query)
    result = run_chromegpt(
        task=user_query,
        model="gpt-3.5-turbo",
        agent="zero-shot",
        headless=False,
        verbose=True,
        continuous=True,
    )

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5000)
