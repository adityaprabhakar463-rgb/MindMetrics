
import shlex
import shutil
import subprocess
import os
from pathlib import Path

def main():
    valid_commands = ("exit","echo","type","pwd","cd")

    while True:
        command = input("$ ")
        tokens = shlex.split(command)

        if command == "exit":
            break

        elif tokens[0] == "echo":
            print(command[5:])

        elif tokens[0] == "type":
            if len(tokens) < 2:
                continue

            cmd = tokens[1]
            if cmd in valid_commands:
                print(f"{command[5:]} is a shell builtin")

            else:
                path = shutil.which(cmd)
                if path:
                    print(f"{cmd} is {path}")
                else:
                    print(f"{cmd}: not found")

        elif tokens[0] == "pwd":
            print(os.getcwd())

        elif tokens[0] == "cd":
            if os.path.isdir(tokens[1]):
                new_dir = tokens[1]
                os.chdir(new_dir)

            elif tokens[1] == "~":
                home_dir = Path.home()
                os.chdir(home_dir)

            else:
                print(f"cd: {tokens[1]}: No such file or directory")

        elif tokens[0] not in valid_commands:
            path = shutil.which(tokens[0])
            if path:
                subprocess.run(tokens)

            else:
                print(f"{tokens[0]}: command not found")

if __name__ == "__main__":
    main()