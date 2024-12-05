# Multi-Agent Example Branch
 
It contains various examples demonstrating the operation of **Multi-Agent systems**.


##  Folder Structure and Executables

- **`Multi_agent/1_Demo_4node`**  
  - Example with a single graph consisting of 5 nodes.  
- **`Multi_agent/2_Two_sub_tree`**  
  - Example with two sub-graphs (one with 4 nodes, another with 3 nodes) and a parent graph combining them into a total of 3 graphs.  

## How to Run

1. Run the **`0.main_chat.py`** file in each folder:
   ```bash
   python 0.main_chat.py
2. API Key is loaded via config_loader.
By default, it uses the path env_path="../../.env" on line 5. Modify the path if necessary.


##  Vewing Full LLM Stream Information

1_Demo_4node
In 0.main_chat.py, uncomment line 48 and enable lines 49–52.

2_Two_Sub_tree
In 0.main_chat.py, enable line 89.


##  Additional Information: Dynamic Prompt Usage

prompt_timp.py
Includes an example of using dynamic prompts. Modify it as needed for your specific use case.
