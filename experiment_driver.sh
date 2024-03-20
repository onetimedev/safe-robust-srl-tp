#!/usr/bin/env bash
echo "---------- Starting Experiments ----------";
# ----- Round 1 ------
printf "\n\n----- Round 1 ------\n\n"
# Section A
python Main.py 250 40 0.2 2 True True True
python Main.py 250 40 0.2 4 True True True
python Main.py 250 40 0.2 6 True True True
python Main.py 250 40 0.2 8 True True True
python Main.py 250 40 0.2 10 True True True

# Section B
python Main.py 250 60 0.2 2 True True True
python Main.py 250 60 0.2 4 True True True
python Main.py 250 60 0.2 6 True True True
python Main.py 250 60 0.2 8 True True True
python Main.py 250 60 0.2 10 True True True

# Section C
python Main.py 250 80 0.2 2 True True True
python Main.py 250 80 0.2 4 True True True
python Main.py 250 80 0.2 6 True True True
python Main.py 250 80 0.2 8 True True True
python Main.py 250 80 0.2 10 True True True

# ----- Round 2 ------
printf "\n\n----- Round 2 ------\n\n"

# Section A
python Main.py 250 40 0.3 2 True True True
python Main.py 250 40 0.3 4 True True True
python Main.py 250 40 0.3 6 True True True
python Main.py 250 40 0.3 8 True True True
python Main.py 250 40 0.3 10 True True True

# Section B
python Main.py 250 60 0.3 2 True True True
python Main.py 250 60 0.3 4 True True True
python Main.py 250 60 0.3 6 True True True
python Main.py 250 60 0.3 8 True True True
python Main.py 250 60 0.3 10 True True True

# Section C
python Main.py 250 80 0.3 2 True True True
python Main.py 250 80 0.3 4 True True True
python Main.py 250 80 0.3 6 True True True
python Main.py 250 80 0.3 8 True True True
python Main.py 250 80 0.3 10 True True True

# ----- Round 3 ------
printf "\n\n----- Round 3 ------\n\n"
# Section A

python Main.py 250 40 0.4 2 True True True
python Main.py 250 40 0.4 4 True True True
python Main.py 250 40 0.4 6 True True True
python Main.py 250 40 0.4 8 True True True
python Main.py 250 40 0.4 10 True True True

# Section B
python Main.py 250 60 0.4 2 True True True
python Main.py 250 60 0.4 4 True True True
python Main.py 250 60 0.4 6 True True True
python Main.py 250 60 0.4 8 True True True
python Main.py 250 60 0.4 10 True True True

# Section C
python Main.py 250 80 0.4 2 True True True
python Main.py 250 80 0.4 4 True True True
python Main.py 250 80 0.4 6 True True True
python Main.py 250 80 0.4 8 True True True
python Main.py 250 80 0.4 10 True True True