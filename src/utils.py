import re

def parse_sections(string_example):
    # Use regular expressions to find sections
    pattern = re.compile(r"#.*?#:")
    matches = list(pattern.finditer(string_example))
    sections = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(string_example)
        section = string_example[start:end].strip()
            
        # Cut off text before the next "/nStep" if it exists
        step_cut = re.search(r'\nStep', section)
        if step_cut:
            section = section[:step_cut.start()]
            
        sections.append(section.strip())
        
    return sections

# def parse_steps(example_string):
#     # Regular expression to match step instructions
#     step_regex = re.compile(r"Step \d+: #([^#]+)#\n([^\n]+(?:\n-(?!Step)[^\n]+)*)", re.MULTILINE)

#     steps_list = []
#     for match in step_regex.finditer(example_string):
#         step_dict = {
#             "step_name": match.group(1).strip(),
#             "step_instruction": match.group(2).strip()
#         }
#         steps_list.append(step_dict)
    
#     return steps_list

def parse_steps(example_string):
    # Extract content inside the first pair of triple backticks
    content_match = re.search(r'```(.*)```', example_string, re.DOTALL)
    if content_match:
        example_string = content_match.group(1).strip()
    
    # Regular expression to match step instructions
    step_regex = re.compile(r"Step (\d+):\s*(?:#([^#]+)#)?\s*(.*?)(?=Step \d+:|$)", re.DOTALL)

    steps_list = []
    for match in step_regex.finditer(example_string):
        step_number = int(match.group(1))
        step_name = match.group(2).strip() if match.group(2) else ""
        step_instruction = match.group(3).strip()
        
        step_dict = {
            "step_number": step_number,
            "step_name": step_name,
            "step_instruction": step_instruction
        }
        steps_list.append(step_dict)
    
    return steps_list