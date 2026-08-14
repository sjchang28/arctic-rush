"""Infrastructure trivia: the startup banner and console logging cadence.

Kept out of `src/config.py` because nothing but `boot.py` and the training
loop's progress lines read it, and 30 lines of ASCII art at the top of the run
configuration buried the settings that matter.
"""

# Console logging cadence. Every episode prints one compact line; the rolling
# averages are only worth a line every so often, otherwise they triple the log
# volume and bury the per-episode results.
LOG_SHORT_AVG_EVERY = 20
LOG_LONG_AVG_EVERY = 100

ASCII_LEELA_BOT = r"""

 _   _ _     ___ _             _              _       _
| | | (_)   |_ _( )_ __ ___   | |    ___  ___| | __ _| |
| |_| | |    | ||/| '_ ` _ \  | |   / _ \/ _ \ |/ _` | |
|  _  | |_   | |  | | | | | | | |__|  __/  __/ | (_| |_|
|_| |_|_( ) |___| |_| |_| |_| |_____\___|\___|_|\__,_(_)
        |/

           ||||||||||||,,
           |WWWWWWWWW|W|||,
           |_________|~WWW||,
            ~-_      ~_  ~WW||,
            __-~---__/ ~_  ~WW|,
        _-~~         ~~-_~_  ~W
  _--~~~~~~~~~~___       ~-~_/
 -                ~~~--_   ~_
|                       ~_   |
|   ____-------___        -_  |
|-~~              ~~--_     - |
 ~| ~--___________     |-_   ~_
   | \`~'/  \`~'_-~~  |  |~-_-
  _-~_~~~    ~~~   _-~  |  |
 ---.--__         ---.-~  |
 | |    -~~-----~~| |    -
 |_|__-~          |_|__-~

"""
