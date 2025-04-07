# DEFENDCLI Artifact Evaluation Guide

This guide explains how to configure and use the DEFENDCLI tool for specific Artifact Evaluation tasks, primarily targeting the E3 (TRACE, THEIA, CADETS) datasets and the A-series attack scenarios (A1, A2, A3).

## Overview

DEFENDCLI is a tool used for analyzing system events and detecting potential threats. This document focuses on how to modify its code to process different datasets and attack scenarios, explains the rule sets used, and describes how to handle the output results for further analysis.

## Prerequisites

1.  **DEFENDCLI Codebase:** You need access to the DEFENDCLI source code.
2.  **Datasets:**
    * For E3 evaluation, ensure the `TRACE`, `THEIA`, or `CADETS` datasets are downloaded and placed in an accessible directory. An example path is `/root/Engagement-3/data/trace`.
    * For A-series evaluation, ensure the `attack_scenario_1.json`, `attack_scenario_2.json`, `attack_scenario_3.json` files are downloaded and placed in an accessible directory. An example path is `/root/`.
    * **Note:** You may need to adjust the file paths mentioned in the instructions below based on your actual data storage location.
3.  **Python Environment:** A Python environment capable of running the DEFENDCLI code is required.

## Usage Instructions

Different code modifications are needed depending on the type of dataset you intend to evaluate.

### 1. Evaluating E3 Datasets (TRACE, THEIA, CADETS)

This configuration is used for processing large-scale system trace data from DARPA Engagement 3.

* **Step 1: Modify the `main()` function**
    * Locate the `main()` function in the code.
    * Find the line of code that sets the dataset directory path.
    * Modify it to point to the root directory of the E3 dataset. For example, for the TRACE dataset:
        ```python
        # Inside the main() function
        directory_path = '/root/Engagement-3/data/trace'
        # !! Important: Please modify '/root/Engagement-3/data/trace' according to your actual data storage path !!
        ```

* **Step 2: Modify the `read_focus_data()` function**
    * Locate the `read_focus_data()` function.
    * Find the line of code used for filtering files, which likely contains a specific filename identifier (e.g., `'ta1-trace-e3-official'`).
    * Modify this identifier based on the specific dataset you are processing (TRACE, THEIA, or CADETS).
        * For **TRACE** (official dataset): Filenames usually contain `'ta1-trace-e3-official'`.
        * For **THEIA**: Change `'ta1-trace-e3-official'` (or another placeholder) in the filter to `'theia'` (or another string that uniquely identifies THEIA data files).
        * For **CADETS**: Change `'ta1-trace-e3-official'` (or another placeholder) in the filter to `'cadets'` (or another string that uniquely identifies CADETS data files).

        Example (assuming original code is for TRACE):
        ```python
        # Inside the read_focus_data() function
        # Original code (processing TRACE):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if
                 not f.endswith('.gz') and 'ta1-trace-e3-official' in f]

        # Modified for THEIA (Example):
        # files = [os.path.join(directory, f) for f in os.listdir(directory) if
        #          not f.endswith('.gz') and 'ta1-theia-e3-official' in f] # <-- Modify the identifier here

        # Modified for CADETS (Example):
        # files = [os.path.join(directory, f) for f in os.listdir(directory) if
        #          not f.endswith('.gz') and 'ta1-cadets-e3-official' in f] # <-- Modify the identifier here
        ```
    * **Note:** Please adjust the string following `in f` according to the actual file naming convention used in your dataset.

### 2. Evaluating Attack Scenarios (A1, A2, A3)

This configuration is used to process specific attack scenario files in JSON format. You will need to run the tool separately for each scenario.

* **Step 1: Modify the `main()` function**
    * Locate the `main()` function.
    * Find the line of code that sets the input file path.
    * Modify it to point to the specific attack scenario JSON file you want to evaluate.

        * For Scenario A1:
            ```python
            # Inside the main() function
            path = '/root/attack_scenario_1.json'
            # !! Important: Please modify '/root/attack_scenario_1.json' according to your actual file storage path !!
            ```
        * For Scenario A2:
            ```python
            # Inside the main() function
            path = '/root/attack_scenario_2.json'
            # !! Important: Please modify '/root/attack_scenario_2.json' according to your actual file storage path !!
            ```
        * For Scenario A3:
            ```python
            # Inside the main() function
            path = '/root/attack_scenario_3.json'
            # !! Important: Please modify '/root/attack_scenario_3.json' according to your actual file storage path !!
            ```
    * Before each run of DEFENDCLI, ensure the `path` variable points to the correct scenario file.

## Rule Sets

For this Artifact Evaluation, the following two rule set files, containing selected attack signatures, are primarily used:

* `cmd_linux.json`: Contains command and behavior signatures relevant to Linux environments.
* `cmd_windows.json`: Contains command and behavior signatures relevant to Windows environments.

These rule sets are used by DEFENDCLI to detect potential malicious activities within the input data.

## Output and GPT Analysis

* **Output File:** After execution, DEFENDCLI will generate a file named `InfoPath.json`. This file contains detected events, associated paths, or other analysis results.
* **GPT Analysis:** The `InfoPath.json` file can be used as input for more advanced analysis, such as interpretation or summarization using large language models (like GPT).
* **Privacy and API:**
    * Due to privacy considerations, this tool does **not** include direct integration with GPT APIs.
    * Users can choose to:
        1.  Manually upload the generated `InfoPath.json` file to a GPT tool that supports file inputs (e.g., GPT-4o).
        2.  Use their own OpenAI API keys to write custom scripts or modify the DEFENDCLI code to call the GPT API for analysis.

---

Hope this more detailed and structured README is helpful!
