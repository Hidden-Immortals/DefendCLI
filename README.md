How to Use DEFENDCLI for Artifact Evaluation

# For DEFENDCLI-E3-TRACE,THEIA,CADETS

-> Go to main():

   modify the code to load the dataset directory_path = '/root/Engagement-3/data/trace'
   
-> Go to read_focus_data():

   files = [os.path.join(directory, f) for f in os.listdir(directory) if
             not f.endswith('.gz') and 'ta1-trace-e3-official' in f]
             
   modify the name of 'trace' to 'theia' or 'cadets'

# For DEFENDCLI-A1,A2,A3

-> Go to main ():

   path = '/root/attack_scenario_1.json'

   modify the file name to 'attack_scenario_1.json', 'attack_scenario_2.json','attack_scenario_3.json'

# Rule Sets

  'cmd_linux.json' and 'cmd_windows.json' are some selected attack signatures used for the artifact evaluation

# GPT Result

  The detector will output a 'InfoPath.json' file, which can be used for GPT analysis.

  Due to privacy issues, we do not provide the use of GPT API here, one can upload the file to GPT4o
  
  or use their own API Keys.

