"""
Example usage of the SWE-RL reward system for scoring code edits and patches.

This example demonstrates how to use the ported SWE-RL reward function
to score code changes against ground truth patches.
"""

import json
from verifiers import swe_rl_reward_func, Rubric
from verifiers.rubrics.swe_rl_utils import FileDiff, FileInfo, FileDiffHeader


def create_example_file_diff():
    """Create an example FileDiff for testing."""
    header = FileDiffHeader(
        file=FileInfo(path="example.py"),
        misc_line=None
    )
    
    file_diff = FileDiff(
        old_file_content="""def hello_world():
    print("Hello world")
    return "Hello"
""",
        new_file_content="""def hello_world():
    print("Hello, World!")
    return "Hello, World!"
""",
        header=header
    )
    
    return file_diff


def main():
    """Demonstrate SWE-RL reward function usage."""
    
    # Create example data
    file_diff = create_example_file_diff()
    
    # Example completion with SEARCH/REPLACE format
    completion = '''
<think>
I need to update the hello_world function to be more professional.
</think>

I'll update the function to use proper punctuation and consistent messaging:

```python
### example.py
<<<<<<< SEARCH
def hello_world():
    print("Hello world")
    return "Hello"
=======
def hello_world():
    print("Hello, World!")
    return "Hello, World!"
>>>>>>> REPLACE
```
'''

    # Ground truth oracle patch
    oracle_patch = """--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def hello_world():
-    print("Hello world")
-    return "Hello"
+    print("Hello, World!")
+    return "Hello, World!"
"""

    # Create file context
    file_context = {
        "example.py": file_diff.old_file_content
    }

    # Create extra info with file diffs and context
    parsed_commit_content = {
        "file_diffs": [file_diff.dict()]
    }
    
    extra_info = {
        "parsed_commit_content": json.dumps(parsed_commit_content),
        "file_context": json.dumps(file_context)
    }

    # Test the reward function directly
    score = swe_rl_reward_func(
        completion=completion,
        answer=oracle_patch,
        info=extra_info
    )
    
    print(f"SWE-RL Score: {score:.4f}")

    # Test using Rubric system
    rubric = Rubric(
        funcs=[swe_rl_reward_func],
        weights=[1.0]
    )
    
    # Note: This would typically be used in an async context
    # scores = await rubric.score_rollout(
    #     prompt="Fix the hello_world function",
    #     completion=completion,
    #     answer=oracle_patch,
    #     state={},
    #     info=extra_info
    # )
    
    print(f"Successfully integrated SWE-RL reward system!")
    print(f"- Created FileDiff utilities for patch handling")
    print(f"- Implemented SEARCH/REPLACE edit parsing")
    print(f"- Added patch similarity scoring using cydifflib")
    print(f"- Integrated with verifiers Rubric system")


if __name__ == "__main__":
    main()