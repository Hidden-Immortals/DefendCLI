# ⚙️ DEFENDCLI Artifact Evaluation Guide ⚙️ 

This guide explains how to configure and use the DEFENDCLI tool for specific Artifact Evaluation tasks, primarily targeting the E3 (TRACE, THEIA, CADETS) datasets and the A-series attack scenarios (A1, A2, A3).

## 📜 Overview

DEFENDCLI is a tool designed for analyzing system events to detect potential threats. This document focuses on:

1.  Modifying the DEFENDCLI codebase to process different datasets and attack scenarios.
2.  Explaining the rule sets used during evaluation.
3.  Describing how to handle the output results for further analysis.

## ✅ Prerequisites

Before you begin, ensure you have the following:

1.  **DEFENDCLI Codebase:** Access to the DEFENDCLI source code.
2.  **Datasets:**
    * **E3 Datasets (TRACE, THEIA, CADETS):**
        * Download the required dataset(s).
        * Place them in an accessible directory.
        * *Example Path:* `/root/Engagement-3/data/trace`
    * **A-Series Attack Scenarios (A1, A2, A3):**
        * Download the `attack_scenario_1.json`, `attack_scenario_2.json`, and `attack_scenario_3.json` files.
        * Place them in an accessible directory.
        * *Example Path:* `/root/`
    > **⚠️ Important:** You **must** adjust the example file paths mentioned in the instructions below to match your actual data storage locations.
3.  **Python Environment:** A working Python environment capable of executing the DEFENDCLI code.

## 🚀 Usage Instructions

Code modifications are required depending on whether you are evaluating large E3 datasets or specific A-series attack scenario files.

---

### 1. Evaluating E3 Datasets (TRACE, THEIA, CADETS)

Use this configuration for processing large-scale system trace data (e.g., from DARPA Engagement 3).

* **Step 1: Modify the `main()` function (Dataset Path)**
    * Locate the `main()` function in the DEFENDCLI source code.
    * Find the line that defines the dataset directory path.
    * Update this path to point to the **root directory** containing the specific E3 dataset files (TRACE, THEIA, or CADETS).

    ```python
    # Inside the main() function

    # Example for TRACE dataset:
    directory_path = '/root/Engagement-3/data/trace'
    # ⬆️ !! Modify this path to your actual dataset location !!
    ```

* **Step 2: Modify the `read_focus_data()` function (File Filtering)**
    * Locate the `read_focus_data()` function.
    * Find the line responsible for filtering files within the specified directory. This line likely contains a string used to identify relevant data files (e.g., `'ta1-trace-e3-official'`).
    * Modify the identifier string based on the dataset you are processing:
        * **TRACE (Official):** Use an identifier like `'ta1-trace-e3-official'`.
        * **THEIA:** Change the identifier to a string unique to THEIA files (e.g., `'ta1-theia-e3-official'`, `'theia'`).
        * **CADETS:** Change the identifier to a string unique to CADETS files (e.g., `'ta1-cadets-e3-official'`, `'cadets'`).

    ```python
    # Inside the read_focus_data() function

    # Original code (Example: processing TRACE):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if
             not f.endswith('.gz') and 'ta1-trace-e3-official' in f]

    # --- Modifications Examples ---

    # Modified for THEIA (Example Identifier):
    # files = [os.path.join(directory, f) for f in os.listdir(directory) if
    #          not f.endswith('.gz') and 'ta1-theia-e3-official' in f] # <-- Modify identifier

    # Modified for CADETS (Example Identifier):
    # files = [os.path.join(directory, f) for f in os.listdir(directory) if
    #          not f.endswith('.gz') and 'ta1-cadets-e3-official' in f] # <-- Modify identifier
    ```
    > **ℹ️ Note:** Carefully check the actual filenames in your downloaded dataset and adjust the identifier string (`'...' in f`) accordingly.

---

### 2. Evaluating Attack Scenarios (A1, A2, A3)

Use this configuration to process individual attack scenario JSON files. You need to configure and run DEFENDCLI separately for *each* scenario (A1, A2, A3).

* **Step 1: Modify the `main()` function (Scenario File Path)**
    * Locate the `main()` function.
    * Find the line that defines the input file path (often named `path`).
    * Modify this path to point directly to the specific attack scenario JSON file you want to evaluate.

    * **For Scenario A1:**
        ```python
        # Inside the main() function
        path = '/root/attack_scenario_1.json'
        # ⬆️ !! Modify this path to your actual file location !!
        ```
    * **For Scenario A2:**
        ```python
        # Inside the main() function
        path = '/root/attack_scenario_2.json'
        # ⬆️ !! Modify this path to your actual file location !!
        ```
    * **For Scenario A3:**
        ```python
        # Inside the main() function
        path = '/root/attack_scenario_3.json'
        # ⬆️ !! Modify this path to your actual file location !!
        ```
    > **⚠️ Important:** Before running DEFENDCLI for a specific scenario, ensure the `path` variable in `main()` correctly points to that scenario's `.json` file. Run the tool once per scenario file.

---

##  Regeln (Rule Sets)

For this Artifact Evaluation, DEFENDCLI primarily utilizes the following rule set files, which contain selected attack signatures:

* `cmd_linux.json`: Contains command and behavior signatures relevant to **Linux** environments.
* `cmd_windows.json`: Contains command and behavior signatures relevant to **Windows** environments.

These files provide the logic DEFENDCLI uses to identify potentially malicious activities within the event data.

## 📊 Output and Further Analysis

* **Output File:** Upon successful execution, DEFENDCLI generates an output file named `InfoPath.json`. This file contains details about detected events, associated provenance paths, or other analysis results based on the activated rules.

* **GPT / LLM Analysis:**
    * The `InfoPath.json` file can serve as structured input for further analysis, interpretation, or summarization using Large Language Models (LLMs) like GPT.
    * **Privacy & API Integration:**
        > Due to privacy considerations and the need for user-specific API keys, this tool **does not** include direct, built-in integration with external LLM APIs (like OpenAI's GPT API).
    * **User Options for LLM Analysis:**
        1.  **Manual Upload:** You can manually upload the generated `InfoPath.json` file to an LLM interface that supports file uploads (e.g., the web interface for GPT-4o or similar models).
        2.  **Custom Scripting:** You can use your own API keys (e.g., OpenAI API key) to write custom scripts that read `InfoPath.json` and interact with the desired LLM API. You could also modify the DEFENDCLI code yourself to add this functionality if preferred.
