import os
current_dir = os.path.dirname(os.path.realpath(__file__))
print(os.path.join(current_dir,"somehting", "*.json"))