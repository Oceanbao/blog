---
title: "SHELL"
date: 2018-12-22T17:01:17-05:00
showDate: true
draft: false
---



```bash
#!/bin/bash
##############################################################################
# SHORTCUTS
##############################################################################


CTRL+A  # move to beginning of line
CTRL+B  # moves backward one character
CTRL+C  # halts the current command
CTRL+D  # deletes one character backward or logs out of current session, similar to exit
CTRL+E  # moves to end of line
CTRL+F  # moves forward one character
CTRL+G  # aborts the current editing command and ring the terminal bell
CTRL+J  # same as RETURN
CTRL+K  # deletes (kill) forward to end of line
CTRL+L  # clears screen and redisplay the line
CTRL+M  # same as RETURN
CTRL+N  # next line in command history
CTRL+O  # same as RETURN, then displays next line in history file
CTRL+P  # previous line in command history
CTRL+R  # searches backward
CTRL+S  # searches forward
CTRL+T  # transposes two characters
CTRL+U  # kills backward from point to the beginning of line
CTRL+V  # makes the next character typed verbatim
CTRL+W  # kills the word behind the cursor
CTRL+X  # lists the possible filename completions of the current word
CTRL+Y  # retrieves (yank) last item killed
CTRL+Z  # stops the current command, resume with fg in the foreground or bg in the background

ALT+B   # moves backward one word
ALT+D   # deletes next word
ALT+F   # moves forward one word

DELETE  # deletes one character backward
!!      # repeats the last command
exit    # logs out of current session


##############################################################################
# BASH BASICS
##############################################################################

env                 # displays all environment variables

echo $SHELL         # displays the shell you're using
echo $BASH_VERSION  # displays bash version

bash                # if you want to use bash (type exit to go back to your previously opened shell)
whereis bash        # finds out where bash is on your system
which bash          # finds out which program is executed as 'bash' (default: /bin/bash, can change across environments)

clear               # clears content on window (hide displayed lines)


##############################################################################
# FILE COMMANDS
##############################################################################


ls                            # lists your files in current directory, ls <dir> to print files in a specific directory
ls -l                         # lists your files in 'long format', which contains the exact size of the file, who owns the file and who has the right to look at it, and when it was last modified
ls -a                         # lists all files, including hidden files (name beginning with '.')
ln -s <filename> <link>       # creates symbolic link to file
touch <filename>              # creates or updates (edit) your file
cat <filename>                # prints file raw content (will not be interpreted)
any_command > <filename>      # '>' is used to perform redirections, it will set any_command's stdout to file instead of "real stdout" (generally /dev/stdout)
more <filename>               # shows the first part of a file (move with space and type q to quit)
head <filename>               # outputs the first lines of file (default: 10 lines)
tail <filename>               # outputs the last lines of file (useful with -f option) (default: 10 lines)
vim <filename>                # opens a file in VIM (VI iMproved) text editor, will create it if it doesn't exist
mv <filename1> <dest>         # moves a file to destination, behavior will change based on 'dest' type (dir: file is placed into dir; file: file will replace dest (tip: useful for renaming))
cp <filename1> <dest>         # copies a file
rm <filename>                 # removes a file
diff <filename1> <filename2>  # compares files, and shows where they differ
wc <filename>                 # tells you how many lines, words and characters there are in a file. Use -lwc (lines, word, character) to ouput only 1 of those informations
chmod -options <filename>     # lets you change the read, write, and execute permissions on your files (more infos: SUID, GUID)
gzip <filename>               # compresses files using gzip algorithm
gunzip <filename>             # uncompresses files compressed by gzip
gzcat <filename>              # lets you look at gzipped file without actually having to gunzip it
lpr <filename>                # prints the file
lpq                           # checks out the printer queue
lprm <jobnumber>              # removes something from the printer queue
genscript                     # converts plain text files into postscript for printing and gives you some options for formatting
dvips <filename>              # prints .dvi files (i.e. files produced by LaTeX)
grep <pattern> <filenames>    # looks for the string in the files
grep -r <pattern> <dir>       # search recursively for pattern in directory


##############################################################################
# DIRECTORY COMMANDS
##############################################################################


mkdir <dirname>  # makes a new directory
cd               # changes to home
cd <dirname>     # changes directory
pwd              # tells you where you currently are


##############################################################################
# SSH, SYSTEM INFO & NETWORK COMMANDS
##############################################################################


ssh user@host            # connects to host as user
ssh -p <port> user@host  # connects to host on specified port as user
ssh-copy-id user@host    # adds your ssh key to host for user to enable a keyed or passwordless login

whoami                   # returns your username
passwd                   # lets you change your password
quota -v                 # shows what your disk quota is
date                     # shows the current date and time
cal                      # shows the month's calendar
uptime                   # shows current uptime
w                        # displays whois online
finger <user>            # displays information about user
uname -a                 # shows kernel information
man <command>            # shows the manual for specified command
df                       # shows disk usage
du <filename>            # shows the disk usage of the files and directories in filename (du -s give only a total)
last <yourUsername>      # lists your last logins
ps -u yourusername       # lists your processes
kill <PID>               # kills the processes with the ID you gave
killall <processname>    # kill all processes with the name
top                      # displays your currently active processes
bg                       # lists stopped or background jobs ; resume a stopped job in the background
fg                       # brings the most recent job in the foreground
fg <job>                 # brings job to the foreground

ping <host>              # pings host and outputs results
whois <domain>           # gets whois information for domain
dig <domain>             # gets DNS information for domain
dig -x <host>            # reverses lookup host
wget <file>              # downloads file


##############################################################################
# VARIABLES
##############################################################################


varname=value                # defines a variable
varname=value command        # defines a variable to be in the environment of a particular subprocess
echo $varname                # checks a variable's value
echo $$                      # prints process ID of the current shell
echo $!                      # prints process ID of the most recently invoked background job
echo $?                      # displays the exit status of the last command
export VARNAME=value         # defines an environment variable (will be available in subprocesses)

array[0]=valA                # how to define an array
array[1]=valB
array[2]=valC
array=([2]=valC [0]=valA [1]=valB)  # another way
array=(valA valB valC)              # and another

${array[i]}                  # displays array's value for this index. If no index is supplied, array element 0 is assumed
${#array[i]}                 # to find out the length of any element in the array
${#array[@]}                 # to find out how many values there are in the array

declare -a                   # the variables are treaded as arrays
declare -f                   # uses function names only
declare -F                   # displays function names without definitions
declare -i                   # the variables are treaded as integers
declare -r                   # makes the variables read-only
declare -x                   # marks the variables for export via the environment

${varname:-word}             # if varname exists and isn't null, return its value; otherwise return word
${varname:=word}             # if varname exists and isn't null, return its value; otherwise set it word and then return its value
${varname:?message}          # if varname exists and isn't null, return its value; otherwise print varname, followed by message and abort the current command or script
${varname:+word}             # if varname exists and isn't null, return word; otherwise return null
${varname:offset:length}     # performs substring expansion. It returns the substring of $varname starting at offset and up to length characters

${variable#pattern}          # if the pattern matches the beginning of the variable's value, delete the shortest part that matches and return the rest
${variable##pattern}         # if the pattern matches the beginning of the variable's value, delete the longest part that matches and return the rest
${variable%pattern}          # if the pattern matches the end of the variable's value, delete the shortest part that matches and return the rest
${variable%%pattern}         # if the pattern matches the end of the variable's value, delete the longest part that matches and return the rest
${variable/pattern/string}   # the longest match to pattern in variable is replaced by string. Only the first match is replaced
${variable//pattern/string}  # the longest match to pattern in variable is replaced by string. All matches are replaced

${#varname}                  # returns the length of the value of the variable as a character string

*(patternlist)               # matches zero or more occurrences of the given patterns
+(patternlist)               # matches one or more occurrences of the given patterns
?(patternlist)               # matches zero or one occurrence of the given patterns
@(patternlist)               # matches exactly one of the given patterns
!(patternlist)               # matches anything except one of the given patterns

$(UNIX command)              # command substitution: runs the command and returns standard output


##############################################################################
# FUNCTIONS
##############################################################################


# The function refers to passed arguments by position (as if they were positional parameters), that is, $1, $2, and so forth.
# $@ is equal to "$1" "$2"... "$N", where N is the number of positional parameters. $# holds the number of positional parameters.


function functname() {
  shell commands
}

unset -f functname  # deletes a function definition
declare -f          # displays all defined functions in your login session


##############################################################################
# FLOW CONTROLS
##############################################################################


statement1 && statement2  # and operator
statement1 || statement2  # or operator

-a                        # and operator inside a test conditional expression
-o                        # or operator inside a test conditional expression

# STRINGS

str1 = str2               # str1 matches str2
str1 != str2              # str1 does not match str2
str1 < str2               # str1 is less than str2 (alphabetically)
str1 > str2               # str1 is greater than str2 (alphabetically)
-n str1                   # str1 is not null (has length greater than 0)
-z str1                   # str1 is null (has length 0)

# FILES

-a file                   # file exists
-d file                   # file exists and is a directory
-e file                   # file exists; same -a
-f file                   # file exists and is a regular file (i.e., not a directory or other special type of file)
-r file                   # you have read permission
-s file                   # file exists and is not empty
-w file                   # your have write permission
-x file                   # you have execute permission on file, or directory search permission if it is a directory
-N file                   # file was modified since it was last read
-O file                   # you own file
-G file                   # file's group ID matches yours (or one of yours, if you are in multiple groups)
file1 -nt file2           # file1 is newer than file2
file1 -ot file2           # file1 is older than file2

# NUMBERS

-lt                       # less than
-le                       # less than or equal
-eq                       # equal
-ge                       # greater than or equal
-gt                       # greater than
-ne                       # not equal

if condition
then
  statements
[elif condition
  then statements...]
[else
  statements]
fi

for x in {1..10}
do
  statements
done

for name [in list]
do
  statements that can use $name
done

for (( initialisation ; ending condition ; update ))
do
  statements...
done

case expression in
  pattern1 )
    statements ;;
  pattern2 )
    statements ;;
esac

select name [in list]
do
  statements that can use $name
done

while condition; do
  statements
done

until condition; do
  statements
done

##############################################################################
# COMMAND-LINE PROCESSING CYCLE
##############################################################################


# The default order for command lookup is functions, followed by built-ins, with scripts and executables last.
# There are three built-ins that you can use to override this order: `command`, `builtin` and `enable`.

command  # removes alias and function lookup. Only built-ins and commands found in the search path are executed
builtin  # looks up only built-in commands, ignoring functions and commands found in PATH
enable   # enables and disables shell built-ins

eval     # takes arguments and run them through the command-line processing steps all over again


##############################################################################
# INPUT/OUTPUT REDIRECTORS
##############################################################################


cmd1|cmd2  # pipe; takes standard output of cmd1 as standard input to cmd2
< file     # takes standard input from file
> file     # directs standard output to file
>> file    # directs standard output to file; append to file if it already exists
>|file     # forces standard output to file even if noclobber is set
n>|file    # forces output to file from file descriptor n even if noclobber is set
<> file    # uses file as both standard input and standard output
n<>file    # uses file as both input and output for file descriptor n
n>file     # directs file descriptor n to file
n<file     # takes file descriptor n from file
n>>file    # directs file description n to file; append to file if it already exists
n>&        # duplicates standard output to file descriptor n
n<&        # duplicates standard input from file descriptor n
n>&m       # file descriptor n is made to be a copy of the output file descriptor
n<&m       # file descriptor n is made to be a copy of the input file descriptor
&>file     # directs standard output and standard error to file
<&-        # closes the standard input
>&-        # closes the standard output
n>&-       # closes the ouput from file descriptor n
n<&-       # closes the input from file descripor n


##############################################################################
# PROCESS HANDLING
##############################################################################


# To suspend a job, type CTRL+Z while it is running. You can also suspend a job with CTRL+Y.
# This is slightly different from CTRL+Z in that the process is only stopped when it attempts to read input from terminal.
# Of course, to interrupt a job, type CTRL+C.

myCommand &  # runs job in the background and prompts back the shell

jobs         # lists all jobs (use with -l to see associated PID)

fg           # brings a background job into the foreground
fg %+        # brings most recently invoked background job
fg %-        # brings second most recently invoked background job
fg %N        # brings job number N
fg %string   # brings job whose command begins with string
fg %?string  # brings job whose command contains string

kill -l      # returns a list of all signals on the system, by name and number
kill PID     # terminates process with specified PID

ps           # prints a line of information about the current running login shell and any processes running under it
ps -a        # selects all processes with a tty except session leaders

trap cmd sig1 sig2  # executes a command when a signal is received by the script
trap "" sig1 sig2   # ignores that signals
trap - sig1 sig2    # resets the action taken when the signal is received to the default

disown <PID|JID>    # removes the process from the list of jobs

wait                # waits until all background jobs have finished


##############################################################################
# TIPS & TRICKS
##############################################################################


# set an alias
cd; nano .bash_profile
> alias gentlenode='ssh admin@gentlenode.com -p 3404'  # add your alias in .bash_profile

# to quickly go to a specific directory
cd; nano .bashrc
> shopt -s cdable_vars
> export websites="/Users/mac/Documents/websites"

source .bashrc
cd $websites


##############################################################################
# DEBUGGING SHELL PROGRAMS
##############################################################################


bash -n scriptname  # don't run commands; check for syntax errors only
set -o noexec       # alternative (set option in script)

bash -v scriptname  # echo commands before running them
set -o verbose      # alternative (set option in script)

bash -x scriptname  # echo commands after command-line processing
set -o xtrace       # alternative (set option in script)

trap 'echo $varname' EXIT  # useful when you want to print out the values of variables at the point that your script exits

function errtrap {
  es=$?
  echo "ERROR line $1: Command exited with status $es."
}

trap 'errtrap $LINENO' ERR  # is run whenever a command in the surrounding script or function exits with non-zero status 

function dbgtrap {
  echo "badvar is $badvar"
}

trap dbgtrap DEBUG  # causes the trap code to be executed before every statement in a function or script
# ...section of code in which the problem occurs...
trap - DEBUG  # turn off the DEBUG trap

function returntrap {
  echo "A return occurred"
}

trap returntrap RETURN  # is executed each time a shell function or a script executed with the . or source commands finishes executing
```



# UNIX

### Quick Reference

**FILE**

```bash
rm -r #delete dir
rm -f # force remove file
cp -r dir1 dir2 # copy 1 to 2 creating dir2 if inexist
ln -s file link # create symbolic link link to file
```

**Process**

```bash
killall proc # kill all processes named proc *
bg # lists stopped or background jobs; 
fg a # brings job a to foreground
```

**File Permissions**

`chmod octal file`  change permissions of file to octal,

**SSH**

```bash
ssh user@host
ssh -p port user@host # connect to host on port port as user
ssh-copy-id user@host # add key to host for user to enable a keyed or passwordless login
```

**Searching**

```bash
grep pattern files # search for pattern in files
grep -r pattern dir # search recursively for pattern in dir
command | grep pattern # search for pattern in output of command
locate file # find all instances of file
```

**Sys info**

```bash
cal, uptime, 
w # who is online
finger user # info about user
uname -a # kernel info
cat /proc /cpuinfo
cat /proc /meminfo
df # disk usage
du # dir space suage
free # mem and swap usage
whereis # possible loc of app
which app # which app will be run by default
```

**Compression**

```bash
tar cf file.tar files # create a tar named file.tar containing files
tar xf file.tar # extract files
tar czf file.tar.gz files # create a tar with Gzip comp
tar xzf file.tar.gz # extract tar using Gzip
tar cjf file.tar.bz2 # creat tar with Bzip2
gzip file # comp file and rename to file.gz (gzip -d)
```

**Network**

```bash
whois domain # get whois info for domain
dig domain # get DNS info for domain
dig -x host # reverse lookup host
wget file

# Examples
wget URL/file.iso
wget --output-document=filename.html example.com
wget --directory-prefix=folder/subfolder example.com
wget --continue example.com/big.file.iso
wget --input list-of-file-urls.txt
wget http://example.com/images/{1..20}.jpg
wget --page-requisites --span-hosts --convert-links --adjust-extension http://example.com/dir/file

# download all MP3 files from a subdir
wget --level==1 --recursive --no-parent --accept mp3,MP3 http://example.com/mp3/
# image
wget --no-directories --recursive --no-clobber --accept jpg,gif,png,jpeg http://example.com/images/
# pdf
wget --mirror --domains=abc.com,files.abc.com,docs.abc.com --accept=pdf http://abc.com/
# Restricted content
wget --refer=http://google.com --user-agent="Mozzila/5.0 Firefox/4.0.1" http://nytimes.com
wget --http-user=labnol --http-password=hello123 URL/secret/file.zip
wget ‐‐cookies=on ‐‐save-cookies cookies.txt ‐‐keep-session-cookies ‐‐post-data ‘user=labnol&password=123’ http://example.com/login.php
wget ‐‐cookies=on ‐‐load-cookies cookies.txt ‐‐keep-session-cookies http://example.com/paywall
# get size of file w/o dl
wget --spider --server-response URL/file.iso
# cat
wget --output-document --quiet URL
# be nice
wget --limit-rate=20k --wait=60 --random-wait --mirror URL

# CURL

# main age
curl http://www.netscape.com/
curl ftp://ftp.funet.fi/README
curl http://www.weirdserver.com:8000/
curl -u username: --key ~/.ssh/id_rsa scp://example.com/~/file.txt


```

**Installation**

```bash
# install from source:
./configure
make
make install
dpkg -i pkg.deb # install a pckg (Debian)
rpm -Uvh pkg.rpm
```



### Find file in current dir with filename, full path printed

find "$(pwd -P)" -name ".zshrc"

### MAKE

`make` reads `Makefile` defining set of tasks to be executed (e.g. compile a program from source code)

```makefile
say_hello: # func name, aka TARGET; DEPENDENCIES follow here
	echo "Hello World" # RECIPE using DEPENDENCIES to make TARGET
	
target: prerequisites
<TAB>recipe

# a target might be binary file depending on prereq (source files) which can also be target depending on other deps

final_target: sub_target final_target.c
	Recipe_to_create_final_target
sub_target: sub_target.c
	Recipe_to_create_sub_target

# To suppress echoing (refer to first example) the acutal CMD, add @
say_hello:
	@echo "Hello World"
generate:
	@echo "Creating empty text files..."
	touch file-{1..10}.txt
clean:
	@echo "Cleaning up..."
	rm *.txt
# the first target is default goal! can be changed by inserting in head
.DEFAULT_GOAL := generate
# this phony target .DEFAULT_GOAL can run only one target at a time, why most makefiles include `all` as target that can call as many targets as needed:
all: say_hello generate
# Before running make, include another special phony target, .PHONY where defined all targets not files - make run its recipe regardless of whether a file with that name exists or what its last mod time is
.PHONY:all say_hello generate clean
all:say_hello generate

# Good practice not to call clean in all or put it as first target. Manually as arg
make clean

# ADVANCED EX
# Variables 
CC := gcc # recursive expanded variable
CC := ${CC}
hello:hello.c
	${CC}hello.c -o hello # recipe expands as below when passed 
gcc hello.c -o hello
# both ${CC} and £(CC) valid ref to call gcc
all:
	@echo ${CC}

# Example: compile all C programs by using var, patterns, func

# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean

CC = gcc                        # compiler to use

LINKERFLAG = -lm

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${BINS}

%: %.o
        @echo "Checking.."
        ${CC} ${LINKERFLAG} $< -o $@

%.o: %.c
        @echo "Creating object.."
        ${CC} -c $<

clean:
        @echo "Cleaning up..."
        rm -rvf *.o ${BINS}
```

Run `make` outputting as expected

- Not necessary target to be file, name for recipe could do 'phony targets'



### SCREEN - Multiple Windows (virtual terminals) in UNIX

- If local machine crashes / lose connection (e.g. SSH), the process or login sessions via `screen` persist - resumed with `screen -r` (may need ARG if multiple detached screens)
  - In some crashes may need to manually "detach" before "Attach" 
    - Find out current screens : `screen -list`
    - Possible states : dead, attached, detached
    - To detach : `screen -D [PID].[machine_arguments]`
    - Resume : `screen -r [PID].[machine_arguments]`
- Cut and paste between screens using CLI + Block copy feature + multiple pages copy/paste
- Detach for resuming after logging out !

```shell
screen # start
Ctrl-a c # create new window/shell
Ctrl-a k # kill current window
w # list all
0-9 # switch window
[ # start copy mode
] # paste copied text
D # power detach and logout
d # detach but keep shell window open
```

- Copy a block : `Ctrl-a [` to move cursor (h, j, k, l) `0` or `^` moves to start and `$` moves to end ; `Ctrl-b` scrolls cursor back one page and `Ctrl-f` forward one ; to set left and right margins of copy `c` and `C` ; spacebar starts selecting text and ends ; to abort copy mode `Ctrl-g`
- Other CMD : `screen unixcommand` in a new window

### 12 ML CLI

- `wget <URL.ext>` 		file retriver for downloading files
- `cat` 	outputting file for preview
- `wc` 		procuding counts, word/line/byte/ etc
- `head / tail`
- `cut -d ',' -f 5 iris.csv` 	**slicing** out sections of line of text; **fifth** col using comma delimiter 
- `uniq` 	unique count of pipeline |
- `awk '/setosa/ { print $0 }' iris.csv`
- `sed / history`

### Globbing

1. list all files anywhere under folder
   - `ls folder/**/*.txt`
2. print results
   - `print -l folder/*/*`
3. wildcard, in crux regex!
   - `*<1-10>.txt`
   - `[a]*.txt` 
   - `(ab|bc)*.txt`
   - `[^cC]*.txt` 
4. qualifiers
   `*(/) # directories`
   `*(.) # regular files`
   `*(L0) # empty files`
   `*(Lk+3) # greater than 3KB`
   `*(mh-1) # modified in last hour`
   `*(om[1,3]) # ordered most to least recently mod and show last 3`
5. show all not having a file

```
print -l folder/*/*(e:'[[ ! -e $REPLY/file ]]':)
	e: ensued by delimited 'cmd'
	$REPLY variable has all file name return one per time
	[[ -e file ]] is cond.expression returns true if file exits, i.e. not true here
```

1. modifier: return file name
   `*.txt(:t)` # tail
   `*.txt(:t:r)` # removed ext
   `*.txt(:e)` # returns ext
2. storing glob as variable
   - `my_file=(folder/data/europe/*.txt([1]))`
   - `print -l ${my_file:h}`

### Magic tabbing

1. Modify .sh file as executable
   - `chmod u+x file.sh`
2. Add to PATH for simple CLI (saved in ~/bin)
   - `export PATH=$PATH:~/bin` # add to end

### Ghostscript: gs -modifying pdf

1. combine pdf files
   - `gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=merged.pdf mine1.pdf mine2.pdf`
2. combine + super-shrink
   - `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/default -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages -dCompressFonts=true -r150 -sOutputFile=output.pdf input.pdf`
3. Convert PDF to PNG
   - `gs -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="Pic-%d.png" input.pdf`

### Print file specially

```shell
- pr -2t filename # 2-column + trimmed
- grep [-v print all UN-match pattern] [-n with line num.] [-l only name of files] [-c counts] [-i caseless]
- ls -l | grep -i "aug" | sort +4n
```

1. print process in full
   - `ps -f`
   - `kill -9 'PID'`
2. run process in backend; bring foreground
   - `ls -a &`
   - `fg jobid # jobs for finding id`
3. check OS of UNIX
   - `uname -a`
4. display current drive info or disk utility
   - `df -h` # human readable for
   - `du -h dir` # dito
5. display process (omitting long command line & include them)
   - `ps -ef`
   - `ps -auxwww`
6. list all processes using a particular file
   - `lsof | grep "filename"`
7. word replacing in file
   - `sed s/old/new/g filename`
8. list only 3rd column of a tap-sep files
   - `cut -f3 filename` # awk is another app

### User MGT

`useradd / usermod / userdel (repace user with group)` # files stored in /etc/passwd & /group

### Files System of UNIX

```shell
/bin  exec binary to all users
/dev  device drivers
/etc  supervisor dir cmds, config files, valid users, ethernet, hosts
/lib  shared library files maybe kernel-related filees
/boot booting Files
/proc processes marked as a file
/var  variable-length files such as log and print
```

```shell
file filename # display file type
head filename
less filename # from end or beginning
more filename # from beginning to back
whereis # show location of files
which filename # location of a file if it is in PATH
```



### Scheduling Task CRON

[article](https://kvz.io/blog/2007/07/29/schedule-tasks-on-linux-using-crontab/)



### YouTube/Video Downloader

Format:

`youtube-dl -F <URL>`

Choosing index

`youtube-dl -f 37 <URL>`

Playlist

`youtube-dl -cit <playlist_url>`

Audio

`youtube-dl -x --audio-format mp3 <video_url>`

[Manual](https://github.com/rg3/youtube-dl/blob/master/README.md#readme)



### KILL PROC LISTENING ON PORT

`lsof -n -i4TCP:8787 | grep LISTEN | awk '{ print $2 }' | xargs kill`





### NETWORKING

`ip addr` shows IPs, MAC, port status etc

`ifconfig` similar output but including packets and bytes count

`route` `-n`  route table

`netstat -n` active connections

`netstat -l -p` listening ports and procs

`tcpdump` CLI Wireshark



### MISC

GITUP SCRIPT (passing read arg)

```shell
read -p "Some prompt msg..." msg </dev/tty
...
git commit -m "$msg"
```



XARGS - map args to cmd

```shell
echo ".\n..\n../.." | xargs ls
```



UNIX MacOS open

```shell
xdg-open index.html
```



STDIN/OUT/ERR

```shell
echo "stdout" >&1
echo "stderr" >&2

# pkg above into sh 'test'
# redirect stout to /dev/null
./test 1>/dev/null

#Redirect all to /dev/null
./test &>/dev/null

# Send output to stdout and any number of additioanl locations with tee
ls && echo "test" | tee file1 file2 file3 && ls

# | pipe stdout to input, > < takes input elsewhere
printf "1\n2\n3" > file && sort < file
```

TEXT EDIT

```shell
# using sed - this doesn
find .txt -exec sed -i '1s/^/task goes here\n/' todo.txt
# add <text> on first 10 lines
sed -i '1,10s/^/<text> /' file
 
 # sub
 cat <(echo "before") text.txt > newfile.txt



# Append/prepend text to file in loops
for file in *.txt ; do
  printf '%s\n' 0a this is text . w | ed -s "$file"
done

# Edit additional text (extra step)
# define text in .txt
```

## SSL/TSL

```shell
openssl genrsa -out example.key 2048

# create CSR (certified signing request) per private key
openssl req -new -key example.key -out example.csr \
-subj "/C=US/ST=TX/L=Dallas/O=Red Hat/OU=IT/CN=test.example.com"

# create certificate per CSR and private key
openssl x509 -req -days 366 -in example.csr \
-signkey example.key -out example.crt
```

