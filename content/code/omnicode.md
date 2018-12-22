---
title: "OMNICODE"
date: 2018-10-22T16:02:02-04:00
showDate: true
tags: ["code", "ml"]
---



## Regex e.g. using sed (stream editor)

`cat /etc/passwd | sed # dump to sed pattern space (i.e. /pattern/action)`
`p (prints line)`
`d (delete line)`
`s/pattern1/pattern2/ (sub 1 with 2)`
examples ( '4, 10d' '4, +5d' '2, 5!d' '1~3d') (sed -n '1, 3p')
`sed 's/old/new/g' (global sub) [-p if subbed, print][-w FILENAME write sub result to file] [-I -i case][M or m empty string]`
Example: matching phone number (chaining)
`sed -e 's/^[[:digit:]]\{3\}/(&)/g' -e 's/)[[:digit:]]\{3\}/&-/g' phone.txt`
`output: (555)555-1212 ...`
`sed '/^daemon/d' # match starting-deamon and delete`
`sed '/sh$/d' # del sh-ending`

- special char: ^ ​$ . (any single char) * (>0 of previous char) [chars]

- Making a BIN exec of any application
`Scripting {#!/bin/bash /Application/LibreOffice.app/Contents/MacOS/soffice "$@"}`

- Put .sh under /usr/local/bin named soffice
`sudo chmod +x /usr/local/bin/soffice`
`convert excel to pdf`
`soffice --headless --convert-to pdf:"filename" /path/.xlsx`





# PYTHON and symlink

### Rename Files

```python
import os, re, glob

for file in glob.glob('*.pdf'):
	new_name = "".join(re.findall('L\d\d|\.pdf', file))
	os.rename(file, new_name)
```



### Learnt about ln cmd to link .sh

`brew unlink python && brew link python`
### this worked to create python with python3
`sudo ln -s /usr/local/bin/python3 /usr/local/bin/python`

### also checkout env variables created in virtualwrapper
WORKON_HOME=/Users/Ocean/.virtualenvs
VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python
VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
VIRTUALENVWRAPPER_PROJECT_FILENAME=.project
VIRTUALENVWRAPPER_WORKON_CD=1
VIRTUALENVWRAPPER_SCRIPT=/usr/local/bin/virtualenvwrapper.sh
VIRTUALENVWRAPPER_HOOK_DIR=/Users/Ocean/.virtualenvs
​	- some are scripted in .bash_profile

creating env dir having exec files + a copy of pip library for installing pkg; omitting name assumes current dir
`virtualenv new_project`

### to change interperter globally
`export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7`

### to activate env
`source new_project/bin/activate` # or ./

### exiting env
`deactivate`

### running env excluding pkg installed globally for keeping clean (default after 1.7) 
`--no-site-packages`

### to snapshot current state of env pkg
`pip freeze > requirements.txt`	# use: `pip install -r requirements.txt`

### VIRTUALENVWRAPPER eases env usage and keeping it one
`pip install virtualenvwrapper`
`export WORKON_HOME=~/Envs` # create default env folder to store 
`source /usr/local/bin/virtualenvwrapper.sh`

### create env
`mkvirtualenv new_project` # created such inside WORKEN_HOME dir i.e. ~/Envs
### init env
`workon new_project`
### combining above two
`mkproject newproject`
### to delete
`rmvirtualenv venv`

### useful cmd
`lsvirtualenv`
`cdvirtualenv` # navigate into dir of currently activated env, so can view its site-packages, e.g.
`cdsitepackages` # like above, but straight to pkg dir
`lssitepackages` # shows contents of pkg dir

### Load Zip file from URL
```python
import requests
import io
import zipfile
def download_extract_zip(url):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield zipinfo.filename, thefile
```

## Basic

```python
# BASE TYPES (immutable)
int 783 0(null) 0b010(binary) 0o642(octal) 0xF3(hexa)
float 9.23 0.0 -1.7e-6
bool True False
str "One\nTwo" "I\'m" """X\t\Y\tZ"""
bytes b"toto\xfe\775"

# CONTAINER TYPES
# Ordered Sequences (fast index access, repeatable values)
list [1, 5, 9] ["x", 11, 8.9] ["mot"]
tuple (1, 5, 9) 11,"y",7.4 ("mot",) # immutable
str bytes # immutable
# key containers (no a priori order, fast key access, each key unique)
dict {"key":"value"} dict(a=3, b=4, k="v")
set {"key1", "key2"} {1, 9, 3, 0} # keys=hasable values (base types, immutables)
frozenset # immutable set

# IDENTIFIERS (for var, fuc, modules, classes.. names)
a...zA...Z_ a...zA...Z_0...9
	# diacritics allowed but should be avoided
    # language keywords forbidden
    # lower/UPPER discrimination
    	# (yes) a toto x7 y_max BigOne
        # (no) 8y and for

# VAR
a=b=c=0 # assignemnt to same value
a, b = b, a
a, *b = seq
*a, b = seq # unpacking of sequence in item and list
del x

# CONVERSION
int("15") -> 15
int("3f", 16) -> 63 # integer number base in 2nd param)
int(15.56) -> 15 
float("-11.25e8") -> -1124000000.0
round(15.56, 1) -> 15.6
bool(x) -> Flase for NULL x, empty container x, None or False x; True else
str(x) -> "..." # repr string of x for display 
chr(64) -> "@" ord('@') -> 64 # code and char
repr(x) -> "...." # literal repr string of x
bytes( [72, 9, 64] ) -> b'H\t@'
list("abc") -> ['a', 'b', 'c']
dict([(3, "three"), (1, "one")]) -> {1:'one', 3:'three'}
set(["one", "two"]) -> {'one', 'two'}
#separator str and sequence of str -> assembled str
':'.join(['toto', '12', 'pswd']) -> 'toto:12:pswd'
# str splitted on whitespsaces -> list of str
"words with spaces".split() -> ['words', 'with', 'spaces']
# str splitted on separator str -> list of str
"1,4,8,2".split(",") -> ['1','4','8','2']
# sequence of one type -> list of another type (via COMP)
[int(x) for x in ('1', '29', '-3')] -> [1,29,3]

# SEQUENCE CONTAINERS INDEXING (lists, tuples, strings, bytes)
# [start:end:step]

# EXCEPTION ON ERROR
# signaling error
raise Exception(...)
finally # block for final processing in all cases
# error processing:
try:
    # normal processing block
except Exception as e:
    # error processing block
```





## PyAutoGui

Move mouse to coordinates of screen

`pyautogui.moveTo(x, y)`

To find out about Pixels screen size - assigning width and height of screen. Anchors to work with to find item on screens of any size.

`x, y = pyautogui.size()`

Similarly, position() will return x and y vaule but instead of MAX height and width, this returns CURRENT location of mouse. Handy in pinpointing where on screen want to clik.

`x, y = pyautogui.position()`

CLICKing right or left button, e.g. first moveTo and Click 

`pyautogui.click()`, `pyautogui.click(button='right')`, `pyautogui.click(200,200)`

`pyautogui.click(x=moveToX, y=moveToY, clicks=num_of_clicks, interval=secs_between_clicks, button='left')`

Using Keyboard + typing text

`pyautogui.typewrite('The text to type')`

Use functional keys, like 'enter', say, to browser Twitter. Instead of passing string, pass a LIST of command, 'enter', or several key names.

`pyautogui.typewrite( [ 'enter' ] )`

`pyautogui.typewrite( ['a', 'b', 'left', 'left', 'X', 'Y'] )`

> this will output "XYab" because it types 'ab', then moves cursor left two spaces, then 'XY' !!!!!!

Sustained key press - they require no button names to LIST, useful for creating programes playing video games.

`KeyDown(keyname)` or `keyUp(keyname)`

**Example: Browsing Twitter**

```python
import pyautogui
from time import sleep

def browse(website):
    global x # assigned later globally
    global y
    
    pyautogui.moveTo(0, y-1) # using Windows Search
    pyautogui.click()
    sleep(1) # wait for loading
    pyautogui.typewrite('Google Chrome')
    sleep(1)
    pyautogui.typewrite(['enter'])
    sleep(5)
    pyautogui.moveTo(297, 63)
    pyautogui.click()
    	# same as pyautogui.click(297, 62)
    pyautogui.typewrite(website)
    pyautogui.typewrite(['enter'])

def tweet(content):
    browse('www.twitter.com')
    global x
    global y
    sleep(5)
    pyautogui.moveTo(x-271, 105) # location gotten via position()
    pyautogui.click()
    sleep(1)
    pyautogui.typewrite(content)
    pyautogui.moveTo(x-666, 492)
    pyautogui.click()

# Get tweet from CLI
theTweet = input('Tweet: ')
x, y = pyautogui.size()
tweet(theTweet)
```

**Automating Boring Stuff**

```python
# Mouse Control
click()
click([x, y])
doubleClick()
rightClick()
moveTo(x, y [, duration = seconds])
moveRel(x_offset, y_offset [, duration = sec]) # relative pixel
dragTo(x, y [, duration = sec]) 
dragRel(x_offset, y_offest, [, duration =sec])
displayMousePosition()
# Keyboard control
typewrite('Text here', [, interval = sec])
press('pageup')
pyautogui.KEYBOARD_KEYS
hotkey('crtl', 'o')
# Image Recognition
	# sudo apt-get scrot
pixel(x, y) # returns RGB tuple
screenshot([filename]) # return PIL/Pillow image obj [saves to file]
locateOnScreen(imageFilename) # returns (left, top, width, height) tuple or None

```

Scrolling 

`pyautogui.scroll(amount_to_scroll, x=moveToX, y=moveToY)`

The full list of key names is in `pyautogui.KEYBOARD_KEYS`

```python
>>> pyautogui.hotkey('ctrl', 'c')  # ctrl-c to copy
>>> pyautogui.hotkey('ctrl', 'v')  # ctrl-v to paste
```

```python
>>> pyautogui.alert('This displays some text with an OK button.')
>>> pyautogui.confirm('This displays text and has an OK and Cancel button.')
'OK'
>>> pyautogui.prompt('This lets the user type in a string and press OK.')
'This is what I typed in.'
```

```python
>>> pyautogui.screenshot()  # returns a Pillow/PIL Image object
<PIL.Image.Image image mode=RGB size=1920x1080 at 0x24C3EF0>
>>> pyautogui.screenshot('foo.png')  # returns a Pillow/PIL Image object, and saves it to a file
<PIL.Image.Image image mode=RGB size=1920x1080 at 0x31AA198>

>>> pyautogui.locateOnScreen('looksLikeThis.png')  # returns (left, top, width, height) of first place it is found
(863, 417, 70, 13)

>>> for i in pyautogui.locateAllOnScreen('looksLikeThis.png')
...
...
(863, 117, 70, 13)
(623, 137, 70, 13)
(853, 577, 70, 13)
(883, 617, 70, 13)
(973, 657, 70, 13)
(933, 877, 70, 13)

>>> list(pyautogui.locateAllOnScreen('looksLikeThis.png'))
[(863, 117, 70, 13), (623, 137, 70, 13), (853, 577, 70, 13), (883, 617, 70, 13), (973, 657, 70, 13), (933, 877, 70, 13)]


>>> pyautogui.locateCenterOnScreen('looksLikeThis.png')  # returns center x and y
(898, 423)

>>> import pyautogui
>>> im = pyautogui.screenshot('saved.png', region=(0,0, 300, 400))

>>> button7location = pyautogui.locateOnScreen('calc7key.png')
>>> button7location
(1416, 562, 50, 41)
>>> button7x, button7y = pyautogui.center(button7location)
>>> button7x, button7y
(1441, 582)
>>> pyautogui.click(button7x, button7y)  # clicks the center of where the 7 button was found

>>> x, y = pyautogui.locateCenterOnScreen('calc7key.png')
>>> pyautogui.click(x, y)
```

There are several “locate” functions. They all start looking at the  top-left corner of the screen (or image) and look to the right and then  down. The arguments can either be a

- `locateOnScreen(image, grayscale=False)` - Returns (left, top, width, height) coordinate of first found instance of the `image` on the screen. Returns None if not found on the screen.
- `locateCenterOnScreen(image, grayscale=False)` - Returns (x, y) coordinates of the center of the first found instance of the `image` on the screen. Returns None if not found on the screen.
- `locateAllOnScreen(image, grayscale=False)` - Returns a generator that yields (left, top, width, height) tuples for where the image is found on the screen.
- `locate(needleImage, haystackImage, grayscale=False)` - Returns (left, top, width, height) coordinate of first found instance of `needleImage` in `haystackImage`. Returns None if not found on the screen.
- `locateAll(needleImage, haystackImage, grayscale=False)` - Returns a generator that yields (left, top, width, height) tuples for where `needleImage` is found in `haystackImage`.

```python
>>> for pos in pyautogui.locateAllOnScreen('someButton.png')
...   print(pos)
...
(1101, 252, 50, 50)
(59, 481, 50, 50)
(1395, 640, 50, 50)
(1838, 676, 50, 50)
>>> list(pyautogui.locateAllOnScreen('someButton.png'))
[(1101, 252, 50, 50), (59, 481, 50, 50), (1395, 640, 50, 50), (1838, 676, 50, 50)]
```

These “locate” functions are fairly expensive; they can take a full second to run. The best way to speed them up is to pass a `region`  argument (a 4-integer tuple of (left, top, width, height)) to only  search a smaller region of the screen instead of the full screen:

```python
>>> import pyautogui
>>> pyautogui.locateOnScreen('someButton.png', region=(0,0, 300, 400))
```

### Pixel Matching

To obtain the RGB color of a pixel in a screenshot, use the Image object’s `getpixel()` method:

```python
>>> import pyautogui
>>> im = pyautogui.screenshot()
>>> im.getpixel((100, 200))
(130, 135, 144)
```

Or as a single function, call the `pixel()` PyAutoGUI function, which is a wrapper for the previous calls:

```python
>>> import pyautogui
>>> pyautogui.pixel(100, 200)
(130, 135, 144)
```

If you just need to verify that a single pixel matches a given pixel, call the `pixelMatchesColor()` function, passing it the X coordinate, Y coordinate, and RGB tuple of the color it represents:

```python
>>> import pyautogui
>>> pyautogui.pixelMatchesColor(100, 200, (130, 135, 144))
True
>>> pyautogui.pixelMatchesColor(100, 200, (0, 0, 0))
False
```

The optional `tolerance` keyword argument specifies how much each of the red, green, and blue values can vary while still matching:

```python
>>> import pyautogui
>>> pyautogui.pixelMatchesColor(100, 200, (130, 135, 144))
True
>>> pyautogui.pixelMatchesColor(100, 200, (140, 125, 134))
False
>>> pyautogui.pixelMatchesColor(100, 200, (140, 125, 134), tolerance=10)
True
```





## ATOM shortcut

===============================================================
S+CMD D : duplicate lines
CTR+CMD ARROW : move lines up/down
CMD+D : select next matched characters
CMD+CTR+G : select all matched characters



## DOCKER
==================================================================
## Basic
docker --version
docker version
docker info

docker container ls
docker container ls --all
docker container ls -aq

docker build -t friendlyhello .  # Create image using this directory's Dockerfile
docker run -p 4000:80 friendlyhello  # Run "friendlyname" mapping port 4000 to 80
docker run -d -p 4000:80 friendlyhello         # Same thing, but in detached mode
docker container ls                                # List all running containers
docker container ls -a             # List all containers, even those not running
docker container stop <hash>           # Gracefully stop the specified container
docker container kill <hash>         # Force shutdown of the specified container
docker container rm <hash>        # Remove specified container from this machine
docker container rm $(docker container ls -a -q)         # Remove all containers
docker image ls -a                             # List all images on this machine
docker image rm <image id>            # Remove specified image from this machine
docker image rm $(docker image ls -a -q)   # Remove all images from this machine
docker login             # Log in this CLI session using your Docker credentials
docker tag <image> username/repository:tag  # Tag <image> for upload to registry
docker push username/repository:tag            # Upload tagged image to registry
docker run username/repository:tag                   # Run image from a registry
docker stack ls                                            # List stacks or apps
docker stack deploy -c <composefile> <appname>  # Run the specified Compose file
docker service ls                 # List running services associated with an app
docker service ps <service>                  # List tasks associated with an app
docker inspect <task or container>                   # Inspect task or container
docker container ls -q                                      # List container IDs
docker stack rm <appname>                             # Tear down an application
docker swarm leave --force      # Take down a single node swarm from the manager
docker-machine create --driver virtualbox myvm1 # Create a VM (Mac, Win7, Linux)
docker-machine create -d hyperv --hyperv-virtual-switch "myswitch" myvm1 # Win10
docker-machine env myvm1                # View basic information about your node
docker-machine ssh myvm1 "docker node ls"         # List the nodes in your swarm
docker-machine ssh myvm1 "docker node inspect <node ID>"        # Inspect a node
docker-machine ssh myvm1 "docker swarm join-token -q worker"   # View join token
docker-machine ssh myvm1   # Open an SSH session with the VM; type "exit" to end
docker node ls                # View nodes in swarm (while logged on to manager)
docker-machine ssh myvm2 "docker swarm leave"  # Make the worker leave the swarm
docker-machine ssh myvm1 "docker swarm leave -f" # Make master leave, kill swarm
docker-machine ls # list VMs, asterisk shows which VM this shell is talking to
docker-machine start myvm1            # Start a VM that is currently not running
docker-machine env myvm1      # show environment variables and command for myvm1
`eval $(docker-machine env myvm1)`         # Mac command to connect shell to myvm1
docker stack deploy -<span class="">c</span> <file> <app>  # Deploy an app; command shell must be set to talk to manager (myvm1), uses loca<span class="">l</span> Compose file
docker-machine scp docker-compose.yml myvm1:~ # Copy file to node's home dir (only required if you use ssh to connect<span class=""> </span>to manager and dep<span class="">l</span>oy the app)
docker-machine ssh myvm1 "docker stack deploy -c <file> <app>"   # Deploy an app using ssh (you must have first copie<span class="">d</span> the Compose file to myvm<span class="">1</span>)
`eval $(docker-machine env -u) `    # Disconnect shell from VMs, use native docker
`docker-machine stop $(docker-machine ls -q)`# Stop all running VMs
`docker-machine rm $(docker-machine ls -q)` # Delete all VMs and their disk images

1) Lists running containers
​	docker ps
​	docker ps -a
​	docker ps -q "quiet or only IDs"

2) Build a docker image using Dockerfile in current dir(.)
​	docker build -t <image_name> .

3) Run docker container (NOTE OPTIONs below able to overwrite Dockerfile specs !! useful at runtime)
​	sodu docker run [OPTION] IMAGE[:TAG|@DIGEST] [COMMAND] [ARG...]

	"sudo docker run -it --name="jupyter" -p 8888:8888 -u="root" -v ~/REPO_Docker/Docker_ML:/home/jupyter oceanbao/machine_learning:base bash"
	
	# Detached vs foreground: if use -d=true and --rm, container removed at exit;
	# foreground attach console to ps stdin/stdout/stderr and even pretend to be TTY
	-a=[] 	: attach to 'STDIN', 'STDOUT', 'STDERR'
	-t 		: allocate a pseudo-tty
	-i 		: keep STDIN open even if not attached
		docker run -a stdin -a stdout -i -t ubuntu /bin/bash
	-it 	: must specify for shell ps
		echo test | docker run -i busybox cat
	# ID: 3 ways as UUID long identifier, UUID short identifier, Name
	--name [NAME]	: specify name, otherwise daemon assign randomly
	--pid=""	: set PID ps Namespace mode 'container:<name|id>': joins another cont's PID namespace / 'host': use the host's PID namespace inside container
	# Network settings
		--dns=[]           : Set custom dns servers for the container
		--network="bridge" : Connect a container to a network
		                      'bridge': create a network stack on the default Docker bridge
		                      'none': no networking
		                      'container:<name|id>': reuse another container's network stack
		                      'host': use the Docker host network stack
		                      '<network-name>|<network-id>': connect to a user-defined network
		--network-alias=[] : Add network-scoped alias for the container
		--add-host=""      : Add a line to /etc/hosts (host:IP)
		--mac-address=""   : Sets the container's Ethernet device's MAC address
		--ip=""            : Sets the container's Ethernet device's IPv4 address
		--ip6=""           : Sets the container's Ethernet device's IPv6 address
		--link-local-ip=[] : Sets one or more container's Ethernet device's link local IPv4/IPv6 addresses
	# Clean up [--rm]: persist by default helping debugging' --rm=true ALSO removes anonymous vol linked with container except those specified 
	# Runtime constraints on resources (see online doc)
	# Overriding Dockerfile image defaults !!
		--entrypoint="" 	:passing clears out any default CMD
		-p=[]		:publish container's port or range of ports to host (-p 8888:8888) (docker port :see mapping)
		--link=""	:add link to another container <name/id>:alias or <name/id>
		-e ""		:HOME=, USER=, HOSTNAME=, PAHT=, 
	# Volume shared filesystems
		-v [host-src:]container-dest[:<options>]
	# User
		-u=""


	docker start <containerID> # rerun after exit; use docker ps -a to see list

4) Pull docker image from Dockerhub
​	docker pull <image_name>

5) List all volumes
​	docker volume ls

6) List
​	docker container ls
​	docker image ls
​	docker volume ls
​	docker network ls

7) Show logs
​	docker logs --follow<container_name>

8) Remove a container
​	docker rm <image_name>

	# remove all with caution!!
	docker rm $(docker ps -aq) 

9) Remove an img
​	docker rmi <image_name>

10) Stop a container
​	docker stop <container_name> "can be hash ID"
​	docker stop $(docker ps -q) "stop all"

11) Shutdown
​	docker kill <container_name>

12) Clean all containers and images
​	docker rm $(docker ps -a -q)
​	docker rmi $(docker images -q)

13) Delete unused resources
​	docker container prune "remove all stopped containers"
​	docker volume prune "remove all unused volumes"
​	docker image prune "remove unused images"
​	docker system prune -a -f --volumes

14) Push image into Dockerhub

​	docker login --username <username> --password <password>
​	docker tage <my_image> <username/my_repo>
​	docker push <username/my_repo>


15) Enter terminal after docker run
​	`docker exec -i -t <container_name> /bin/sh`

16) Commit as New Image from a container's changes
​	`docker commit [OPTIONS] CONTAINER [REPO[:TAG]]
​	-c (apply Dockerfile to the created image)`
​		The --change option will apply Dockerfile instructions to the image that is created. 
​		Supported Dockerfile instructions: 
​		`CMD|ENTRYPOINT|ENV|EXPOSE|LABEL|ONBUILD|USER|VOLUME|WORKD`
​		Example:

```bash
$ docker inspect -f "{{ .Config.Env }}" c3f279d17e0a
		> [HOME=/ PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin]
$ docker commit --change "ENV DEBUG true" c3f279d17e0a  svendowideit/testimage:version3
		> f5283438590d
$ docker inspect -f "{{ .Config.Env }}" f5283438590d
		> [HOME=/ PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin DEBUG=true]
# Commit with new CMD / EXPOSE
$ docker commit --change='CMD ["apachectl", "-DFOREGROUND"]' -c "EXPOSE 80" c3f279d17e0a  svendowideit/testimage:version4
	-m (commit message)
	-p (pause container during commit)
```

17) Connecting Containers (--link)

```bash
docker pull redis
docker run -d --name redis_server redis
docker run -it --name redis_client1 --link redis_server:redis redis bash
docker run --rm -it --link myredis:redis redis bash
> redis-cli -h redis -p 6379
> set key value
> get key
docker run --rm --volumes-from myreids -v $(pwd)/backup:/backup debian cp /data/dum.rdb /backup/ 
# -v mount known dir on host and --volumes-from to connect new container to Redis db folder

# cat /etc/hosts
> 127.0.0.1		localhost
> ...
> 172.17.0.2 	redis <container ID> redis_server

# ping redis

# redis-cli -h redis
# redis:6379> PING
> PONG
# redis:6379> set myname ocean

docker run -it --name redis_client2 --link redis_server:redis redis bash
# redis-cli -h redis
# redis:6379> get myname
> ocean

```

### Play with Docker tutorial

```bash
docker container run \
--detach \
--name mydb \
-e MYSQL_ROOT_PASSWORD=my-secret-pw \
mysql:latest

docker container logs mydb
docker container top mydb

docker container run \
--detach \
--publish 80:80 \
--name linux_tweet_app \
--mount type=bind,source="$(pwd)",target=/usr/share/nginx/html \
$DOCKERID/linux_tweet_app:1.0


```

```dockerfile
 FROM nginx:latest

 COPY index.html /usr/share/nginx/html
 COPY linux.png /usr/share/nginx/html

 EXPOSE 80 443     

 CMD ["nginx", "-g", "daemon off;"]
```



### Resources Usage

`docker ps --format "{{.Names}}" | xargs docker stats`

## O'reilly DOCKER

```bash

```





# XPATH

---
Xpath is a language for addressing parts of an XML document - 1.0
```shell
- element nodes <p>...</p> or tag
- attribute nodes href="page.html"
- text nodes "Some Title" NOT ELEMENTS
- comment nodes <!-- comment... -->

- //html/head/title = $$$<title>....</title>$$$
- //meta/@content = <meta content=$$$"text...stuff"$$$ http-equiv="content-type">
- //div/div[@class="second"] = $$$<div class="second"> everything in side </div>$$$
- //div/a/text() = ... <a href="page3.html">$autre lien$</a> ....
- //div/a/@href =  ... <a href=$$$"page3.html"$$$>autre lien</a> ....
```


## /step1/step2/... each step: AXIS :: NODETEST [PREDICATE]* WHITESPACE NO MATTER
```shell
/html/head/title = /child:: html /child:: head /child:: title
//meta/@content = /descendant-or-self:: node()/child:: meta/attribute:: content
//div/div[@class="second"] = /descend-or-self::node() /child::div /child::div [ attribute::class = "second"]
//body//*[self::ul or self::ol]//li :multiple node names testing, middle-location
```

**AXES = directions**
self = context
parent, child = direct hop
ancestor, ancestor-or-self, descendant, descendant-or-self, = multi-hop
following, following-sibling, preceding, preceding-sibling = document order
attribute, namespace = non-element**

**PREDICATE nested:**

```shell
//div[p[a/@href="sample.html"]]
* = all element nodes bar text/attribute, etc .//* != .//node()
	@* = attribute::*	all attribute nodes
// = /descendant-or-self::node()/
	. = self::node() 	the context node
.. = parent::node()
```

**ATTRIBUTE @**
```
//@id
//BBB[@id]
//BBB[@name]
//BBB[not(@*)]
ATTRIBUTE VALUES
//BBB[@id='b1']
//BBB[@name='bbb']
//BBB[normalize-space(@name)='bbb']
```

**NODE COUNTING**
```
//*[count(BBB)=2]
//*[count(*)=2]
count(//@*)
```
**NAMING ELEMENT**
```
//*[name()='BBB']
//*[starts-with(name(), 'B')]
//*[contains(name(), 'C')]
```
**COMBINING**
```
//AAA/EEE | //DDD/CCC | /AAA
```
EXAMPLES:

```
//div[ a [text() = "link"]] 	
​	:div having a tag with text 'link' = //div[ a/text()="link"]
//a[starts-with(@href, "https")]	
​	:all a tag with href starting with 'https'
//p[ a/@href="https://scrapy.org" ]
​	:value of href attribute from all a tag
//div[@id='footer']/preceding-sibling::text()[1]	
​	:first text node before div footer
//p[text()="Footer text"]/..	
​	:select parent of <p> embedding 'Footer text'
//*[p/text()="Footer text"]		
​	:from all tag, <p> child having text "Footer text"
//li//@href 	
​	:all of href attributes under li, return its value
//li[re:test(@class, "item-\d$")]//@href 	
​	:like above, but only class attribute end in "item-\d$"
string(/html/head/title) 	
​	:returns string repr of elements
```

**VARIABLES**
```
//div[@id=$val]/a/text(),  val='images'	element 'id' attr having 'images'
//div[count(a)=$cnt]/@id,  cnt=5 	find 'id' attr of <div> having 5 <a> children
```
**TRICK**
```
.//text()	collection of text elements as node-set
//a[contains(.//text(), 'target')] = string ONLY first element
//a[contains(., 'target')] = all of <a> tag having 'target'; '.' means current node !!
```
**SPECIFIC CLASS SELECTION**
```
//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')]
```
**CSS + XPATH**
```
css(".content").xpath('@class').extract()
```

**SCRAPY TIP**

 - removing namespaces - bare element names to write more simple XPaths	`response.selector.remove_namespaces()`
 - [] or list() return from response.xpath('//link') : returns [<Selector xpath='//link' data=u'<link xmlns="http://www.w3.org/2005/Atom">,..]

```shell
>>> from scrapy import Selector
>>> sel = Selector(text='
	<div class="hero shout"><time datetime="2014-07-23 19:00">Special date</time></div>')
>>> sel.css('.shout').xpath('./time/@datetime').extract()
[u'2014-07-23 19:00']

>>> response.xpath('//a[contains(@href, "image")]/text()').re(r'Name:\s*(.*)')
[u'My image 1',
 u'My image 2',
 u'My image 3',
 u'My image 4',
 u'My image 5']

Use it to extract just the first matching string:
>>> response.xpath('//a[contains(@href, "image")]/text()').re_first(r'Name:\s*(.*)')
u'My image 1'

//a[contains(@href, "image")]/img/@src 		value 'image.jpg' of <a> having href="image"
```

# HUGO
---
Quickstart
### Create New Site in folder quickstart
hugo new site quickstart
### Add Theme 
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
### Edit config.toml to add Ananake Theme
echo 'theme = "ananke"' >> config.toml
### Add Content
hugo new posts/my-first-post.md
### Start server with drafts enabled
hugo server -D
### Customize Theme configure config.toml

Install and Use Themes
### Cloning entire Hugo Theme repo on locally
git clone --depth 1 --recursive https://github.com/gohugoio/hugoThemes.git themes
### Before using a theme, remove .git folder in that theme's root folder
### Single Theme
cd themes
git clone URL_TO_THEME
### Apply theme: Hugo applies decided theme first then applies anthing local, allowing easier customisation while retinaing compatibility with upstream version of theme
	- change theme via CLI
	`hugo -t themename`
	- or add when servering
	`hugo server -t themename`
	- config File method: add theme directly to site config file
	theme: themename

### GitHub Hosting: 2 types of Pages - User/Org Page and Project Pages
1. User Page: Content from 'master' branch will be used to publish Page site
    - `create <PROJECT>` repo on GitHub e.g. blog having Hugo's content and other source files
    - `create <USERNAME>.github.io repo`, where lie fully rendered version of Hugo website
    - `git clone <PROJECT_URL> && cd <PROJECT>`
    -  `hugo server -t <theme>`
    -  `inspect and rm -rf public `
    -  `git submodule add -b master git@github.com:<username>/<username>.github.io.git public`
		-  creating a git submodule
		-  when run hugo CLI to build site to public folder, it will have a different remote origin (i.e. hosted GitHub repo)
		-  auto steps with script deploy.sh 
		`./deploy.sh` "commit message" to update username.github.io

2. Project Pages
	- ensure baseURL key-value in site configuration reflects full URL of GitHub pages repo
	- e.g. <username>.github.io/<project>/
	- Deploy from /docs folder on master branch
	- change Hugo publish directory in site's config.toml and config.yaml
	- publishDir = "docs"
	- publishDir: docs
	- after running hugo, push master branch to remote repo and choose docs/ folder as source
	- Settings (project) -> GitHub Pages -> Source: master branch /docs folder
	- docs/ option is simplest but need setting a publish dir in site config; 
Deploy from gh-pages branch 
	- or point to gh-pages branch, more complex but keeps source and rendered site separate + using default public folder
	`echo "public" >> .gitignore`
    `git checkout --orphan gh-pages`
    `git reset --hard`
    `git commit --allow-empty -m "Init gh-pages branch"`
    `git push upstream gh-pages`
    `git checkout master`

	- Build and Deploy
	`rm -rf public`
	`git worktree add -B gh-pages public upstream/gh-pages`

	- regenerate site usng hugo and commit fiels on gh-pages branch
	`hugo`
	`cd public && git add --all && git commit -m "Publishing to gh-pages" && cd ..`
	`git push upstream gh-pages`
	- set gh-pages as Publish Branch
	- Settings -> GitHub Pages -> Source: select 'gh-pages branch' -> Save
	- refer to auto-script as publish_to_ghpages.sh
	- this will abort if there are pending changes in working dir and ensure all existing output files are removed. Adjust script to need: include final push to remote repo if no need to take a look or add echo domainname.com >> CNAME if set up for customised domain