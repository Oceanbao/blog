---
title: "LinuxHacker"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

[toc]

# Text Wrangling

```bash
cat /etc/snort/snort.conf

nl /etc/snort.conf 

# example
tail -n+507 /etc/snort/snort.conf | head -n 6

sed s/mysql/MySQL/g /etc/snort/snort.conf > snort2.conf
```



# Network

```bash
curl https://ipinfo.io/ip

ifconfig
iwconfig

ifconfig eth0 192.168.181.115

ifconfig etho0 192.168.181.115 netmask 255.255.0.0 broadcast 192.168.1.255

ifconfig etho down
ifconfig etho hw ether 00:11:22:33:44:55
ifconfig etho up

dhclient etho

dig hackers-arise.com ns
dig hackers-arise.com mx
vim /etc/resolv.conf 
vim /etc/hosts

ss # new netstat
https://www.tecmint.com/ss-command-examples-in-linux/

ip addr show
ip route show
mtr baidu.com
printf 'HEAD / HTTP/1.1\r\nHost: www.baidu.com\r\n\r\n' | nc www.baidu.com

```



# APP

```bash
apt-cache search snort

cat /etc/apt/sources.list
deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main 
deb-src http://ppa.launchpad.net/weupd8team/java/ubuntu precise main

git clone https://www.github.com/balle/bluediving.git
```



# FILE

**USER**

```bash
compgen -u
less /etc/passwd

cut -d: -f1 /etc/passwd
getent passwd | awk -F:'{print $1}'

who
# The second column will give you what type of connection it is: if it’s represented with a “:X” where X is a number, it means it is using a Graphical User Interface (GUI) or Desktop session such as Gnome, XDE, etc; if it says “pts/X” where X is a number, it means it’s a connection made through SSH protocol (command line).

# list all cmd available
compgen -c

chown $user $file
chgrp $group $app

chmod 777 FILE
chmod u-w FILE # user minus write

```



`drwxr-xr-x 5 root root 4096 Dec 5 10:47 charsets

- file type
- Permission of owner, group, users
- number of links
- size



Exploit SUID files

```bash
find / -user root -perm -4000
```



**rsync**

https://www.tecmint.com/rsync-local-remote-file-synchronization-commands/

https://www.tecmint.com/sync-files-using-rsync-with-non-standard-ssh-port/