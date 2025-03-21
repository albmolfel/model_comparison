import json
import pandas as pd

# Load JSON data (replace with actual file path)
with open("results_1.json", "r") as file:
    data = json.load(file)

# Dictionary to store structured results
structured_data = {}

# Iterate through experiments
for exp_name, exp_data in data.items():
    for model_name, model_data in exp_data["results"].items():
        success_rate = model_data["success_rate"]
        execution_rate = model_data["execution_rate"]
        redefinition_rate = model_data["redefinition_rate"]

        for iteration in model_data.get("iterations", []):
            iteration_num = iteration["iteration"]
            success = int(iteration["success"])
            retries = iteration["retries"] +1
            output_values = [
                out[list(out.keys())[0]]["output"]
                for out in iteration["outputs"]
            ]  # Extract all output values

            # Flatten outputs into a single string (if multiple tries)
            output_str = "; ".join(map(str, output_values))

            # Unique key for model + iteration
            key = (model_name, iteration_num)

            # Ensure key exists in the data dictionary
            if key not in structured_data:
                structured_data[key] = {
                    "Model": model_name,
                    "Iteration": iteration_num,
                    "Success Rate": success_rate,
                    "Execution Rate": execution_rate,
                    "Redefinition Rate": redefinition_rate,
                }

            # Store values under corresponding experiment
            structured_data[key][f"{exp_name} Success"] = success
            # structured_data[key][f"{exp_name} Output"] = output_str
            structured_data[key][f"{exp_name} Run"] = retries

# Convert to DataFrame
df = pd.DataFrame(list(structured_data.values()))

# Save to Excel
df.to_excel("experiment_results.xlsx", index=False)

# Save to CSV (optional)
df.to_csv("experiment_results.csv", index=False)

print("Export complete!")
