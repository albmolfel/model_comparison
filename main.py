import ollama
import json
from typing import List, Dict, Any
import re
import sys
import traceback
import builtins
from io import StringIO
from rich.table import Table
from rich.live import Live
from rich.console import Console
import pandas as pd
import argparse
import signal


INTRO = """
Generate a python script in Markdown format (like this ```python```) for the following query.
Only write code, no commentary or explanation.
The only allowed third party modules are numpy, pandas and sklearn.
Using any other third party module or opening files is strictly prohibited.",
Print the desired result to standard output. 
If the query expects a numeric result, make sure to print the number.\n
"""

class Experiment:
    def __init__(self, name: str, prompt: str, expected_output: List[Any], range: bool = False, input_data: Any = None):
        self.name = name
        self.prompt = prompt
        self.expected_output = expected_output
        self.input_data = input_data
        self.range = range

    def evaluate(self, output: str) -> bool:
        if self.range:
            match = re.search(r'-?\d+(\.\d+)?', output)
            if not match:
                return False
            return self.expected_output[0] <= float(match.group()) <= self.expected_output[1]
        else:
            return self.expected_output[0] in output


class CodeExecutor:
    @staticmethod
    def execute(result: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 seconds timeout
        pattern = r'```(?:\w+)?\s*\n(.*?)(?=```)```'
        code = re.findall(pattern, result, re.DOTALL | re.MULTILINE)
        code = "\n".join([r for r in code]) if code else None
        
        exec_globals = {
            '__builtins__': {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith('__')}
        }
        
        exec_globals['__builtins__']['__import__'] = __import__

        for var_name, var_value in input_data.items():
            exec_globals[var_name] = var_value

        prev_stdout, prev_stderr = sys.stdout, sys.stderr
        captured_stdout, captured_stderr = StringIO(), StringIO()
        sys.stdout, sys.stderr = captured_stdout, captured_stderr
        timeout = False
        try:
            exec(code, exec_globals)
        except TimeoutError as e:
            captured_stderr.write(str(e))
            timeout = True
        except Exception as e:
            traceback_str = traceback.format_exc()
            captured_stderr.write(traceback_str)
        finally:
            sys.stdout, sys.stderr = prev_stdout, prev_stderr
            signal.alarm(0)
        
        output = captured_stdout.getvalue().strip()
        error = captured_stderr.getvalue().strip()

        executable = (error == "" or timeout)

        return {"output": output, "error": error, "executable": executable, "code": code}



class ModelEvaluator:
    def __init__(self, models: List[str], experiments: List[Experiment], iterations: int = 5, retries: int = 3):
        self.models = models
        self.experiments = {exp.name: exp for exp in experiments}
        self.retries = retries
        self.iterations = iterations
        self.results = {exp.name: {"prompt": "", "results": {}} for exp in experiments}

    def create_table(self):
        table = Table(expand=True)
        table.add_column(f"Experiment (retries = {self.retries}, iterations = {self.iterations})", justify="left")
        for model in self.models:
            table.add_column(model, justify="center")
        
        for experiment_name, experiment in self.experiments.items():
            row = [experiment_name]
            for model in self.models:
                if model in self.results[experiment_name]["results"]:
                    success_rate = round(self.results[experiment_name]["results"][model]["success_rate"], 2)
                    execution_rate = round(self.results[experiment_name]["results"][model]["execution_rate"], 2)
                    row.append(f"{execution_rate}/{success_rate}")
                else:
                    row.append("⏳")
            table.add_row(*row)
        
        return table

    def run(self):
        with Live(self.create_table(), refresh_per_second=2) as live:
            for experiment_name, experiment in self.experiments.items():
                exp_result = {"prompt": experiment.prompt.splitlines(), "expected_output": experiment.expected_output, "results": {}}

                for model in self.models:
                    model_result = {"success_rate": 0.0, "execution_rate": 0.0, "redefinition_rate": 0.0, "iterations": []}
                    total_runs = 0
                    total_executions = 0
                    total_successes = 0
                    total_redefinitions = 0
                    for x in range(self.iterations):
                        iteration_result = {f"iteration": x, "success": False, "errors": 0, "outputs": []}

                        for i in range(self.retries):
                            code = self.generate_code(model, experiment.prompt)
                            exec_result = CodeExecutor.execute(code, experiment.input_data)
                            success = experiment.evaluate(exec_result["output"])
                            error = exec_result["error"] != ""
                            redefinition = any(f"{variable_name} = " in code for variable_name, var in experiment.input_data.items())
                            total_runs += 1
                            total_executions += not error or success # in some cases, warning look like errors
                            total_successes += success
                            total_redefinitions += redefinition

                            iteration_result["success"] = success

                            iteration_result["outputs"].append({
                                f"try {i}": {
                                    "code": None if exec_result["code"] is None else exec_result["code"].splitlines(),
                                    "output": exec_result["output"],
                                    "executable": exec_result["executable"],
                                    "successful": success,
                                    "error": exec_result["error"].split("\n")[-1] if error != "" else None,
                                    "redefinition": redefinition
                                }
                            })

                            if not error:
                                break
                            iteration_result["errors"] += 1
                        model_result["iterations"].append(iteration_result)
                    model_result["success_rate"] = total_successes / total_runs
                    model_result["execution_rate"] = total_executions / total_runs
                    model_result["redefinition_rate"] = total_redefinitions / total_runs
                    exp_result["results"][model] = model_result
                    self.results[experiment_name] = exp_result
                    live.update(self.create_table())
                    self.save_results()

    def generate_code(self, model: str, prompt: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": INTRO}, 
                {"role": "user", "content": prompt},
            ]
        )
        return response["message"]["content"]

    def save_results(self, filename: str = "results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)


def tables_summary(tables: Dict[str, pd.DataFrame], serialize: bool = False, head: int = 3) -> str:

    table_strings = ""

    for variable_name, df in tables.items():
        if serialize:
            table_strings += f'<dimensions="{df.shape[0]}x{df.shape[1]}">\n'
            table_strings += f"{variable_name} = \n"
            table_strings += df.head(head).to_csv(index=False)
            table_strings += "\n"
        else:
            table_strings += f"{variable_name} = \n"
            buffer = StringIO()
            df.head(head).to_string(buffer, index=False, col_space=15)
            table_strings += buffer.getvalue()
            table_strings += "\n\n"

    return table_strings

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model evaluation experiments.")
    parser.add_argument("-r", "--retries", type=int, default=3, help="Max number of retries")
    parser.add_argument("-i", "--iterations", type=int, default=3, help="Number of iterations")
    args = parser.parse_args()

    models = [
        "dolphin3:8b",
        "deepseek-coder:6.7b",
        "codegemma:7b",
        "codellama:13b",
        "qwen2.5-coder:14b",
        "mistral-nemo:12b"
        "llava:13b",
        "granite-code:8b",
        "deepseek-coder-v2:16b"
    ]
    
    revenue_cost = pd.read_csv("test_data/revenue_cost.csv")
    revenue_cost_explanation = f"""
Assume you have a global variable called 'revenue_cost' which is a pandas dataframe that looks like this:
{tables_summary({"revenue_cost": revenue_cost}, serialize=True)}
Do NOT define 'revenue_cost' again
    """
    pl = pd.read_csv("test_data/pl.csv")
    pl_explanation = f"""
Assume you have a global variable called 'pl' which is a pandas dataframe, extracted from a spreadsheet that looks like this:
{tables_summary({"pl": pl}, serialize=False, head=100)}
Do NOT define 'pl' again
    """

    covariance = pd.read_csv("test_data/covariance.csv", header=None).values
    asset_weights = pd.read_csv("test_data/asset_weights.csv", header=None).values.flatten()
    portofolio_risk_explanation = f"""
Assume you have a global variable called 'covariance' which the covariance matrix of all the assets and an array 'asset_weights' with the weight of each one.
Do NOT define 'covariance' or 'asset_weight' again
    """

    break_even = pd.read_csv("test_data/break_even.csv")
    break_even_explanation = f"""
Assume you have a global variable called 'break_even' which is a pandas dataframe that looks like this:
{tables_summary({"break_even": break_even}, serialize=True)}
Do NOT define 'break_even' again
    """

    

    experiments = [
        Experiment(
            "Linear Regression Forecast",
            "Get the profit for index and then predict the next value using linear regression\n"
            f"{revenue_cost_explanation}", 
            [84110, 84500], range=True,
            input_data={"revenue_cost": revenue_cost}),
        Experiment(
            "Dataframe handling 1",
            "Get the average revenue for the indeces with at least 800 in revenue and between 500 and 600 in cost\n"
            f"{revenue_cost_explanation}", 
            [36395, 36396], range=True,
            input_data={"revenue_cost": revenue_cost}),
        Experiment(
            "Dataframe handling 2",
            f"Print the indeces (sorted and separated by commas) with the highest revenue-to-cost ratio, but only consider those where the revenue is above 1000 and costs are below 400.\n"
            f"{revenue_cost_explanation}", 
            ["292"], 
            input_data={"revenue_cost": revenue_cost}),
        Experiment(
            "Dataframe handling 3",
            "Print the indices (sorted and separated by commas) where revenue has been consistently decreasing for the last three and where the average cost is above 200.\n"
            f"{revenue_cost_explanation}", 
            ["9,10,51"], 
            input_data={"revenue_cost": revenue_cost}),
        Experiment( 
            "Find and Operate 1",
            "Get the gross profit margin (xx.xx%)\n"
            f"{pl_explanation}", 
            ["40.00%"], 
            input_data={"pl": pl}),
        Experiment( 
            "Find and Operate 2",
            "Get the expense ratio (xx.xx%)\n"
            f"{pl_explanation}", 
            ["16.67%"], 
            input_data={"pl": pl}),
        Experiment(
            "Nested Operations",
            "Get the portofolio risk (standard deviation)\n"
            f"{portofolio_risk_explanation}",
            [0.226, 0.227], range=True,
            input_data={"covariance": covariance, "asset_weights": asset_weights}),
        Experiment(
            "Gradient",
            "Get the sum of units needed to break even in each row, don't use the formula, use something similar to gradient descent to find the minimum units.\n"
            f"{break_even_explanation}",
            ["1477"],
            input_data={"break_even": break_even}),
    ]
    evaluator = ModelEvaluator(models, experiments, iterations=args.iterations, retries=args.retries)
    evaluator.run()