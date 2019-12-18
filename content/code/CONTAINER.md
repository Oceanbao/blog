---
title: "CONTAINER"
date: 2019-12-18T15:52:43+08:00
showDate: true
draft: false
---

# 100KB Image from SCRATCH

### Dockerfile has 2 build stages

1. **Builder** - containing all build dependencies including source, libs, tools, etc
2. **Final Image** - containing **binary** and **run-time** dependencies (config files, certificates, and dynmacically linked libs)



### `hello.c`

```c
#include <stdio.h>

int main(void) {
  puts("Hello World\n");
}
```

### `Makefile`

```makefile
hello: hello.c
	gcc -o $@ $< -static
```

### `.dockerignore`

```dockerfile
# preventing host binary from sneaking into build
hello
```

### `Dockerfile`

```dockerfile
FROM alphine:latest as builder
WORKDIR /build
RUN apk update && \
		apk add gcc make libc-dev
COPY . ./
RUN make

FROM scratch
WORKDIR /
COPY --from=builder /build/hello .
CMD ["/hello"]
```



### Build and Run

```shell
docker build . -t hello-scratch | tail -n 1
docker run hello-scratch
docker image | grep hello-scratch | egrep -o '[^ ]+`

mkdir hello-scratch && cd hello-scratch
docker save hello-scratch | tar -x
ls
tar -tf 5599.../layer.tar
```



# Best Practice of Dockerfile

[5 Tips](https://dev.to/azure/improve-your-dockerfile-best-practices-5ll)



# Life and Death of Container

### Checking event

```shell
t0=$(date "+%Y-%m-%dT%H:%M:%S")
docker run --name=ephemeral -t lherrera/cowsay 'I am ephemeral'
t1=$(date "+%Y-%m-%dT%H:%M:%S")
docker events --since $t0 --until $t1

# all container data persist till 'docker rm' thus exportable FS as tar
docker export -o ephemeral.tar ephemeral
tar tvf ephemeral.tar
tar xvf ephemeral.tar var/tmp/legacy
cat var/tmp/legacy

# docker export won't preserve history but shrink to single layer
docker history lherrera/cowsay
docker import ephemeral.tar lherrera/cowsay:2.0
docker history lherrera/cowsay:2.0

# short-running foreground containers data can pile up hence
docker run --rm ...

# pause container to backup or stop slow running
docker run -d -p 80:80 --name web nginx:alpine
docker pause web
docker ps
docker export -o web.tar web
curl -v 0.0.0.0:80
# in another terminal type 'docker unpause web'

# OOM inside container and on container run itself WHY?
docker run -it -m 4m ubuntu:14.04 bash
root@cffc126297e2:/# python3 -c 'open("/dev/zero").read(5*1024*1024)'
>>> killed

docker run  -m 4m ubuntu:14.04 python3 -c 'open("/dev/zero").read(5*1024*1024)'
docker ps -a
# Exited (137)

# Docker 1.12 added health check
docker run --name=web -d \
    --health-cmd='stat /etc/nginx/nginx.conf || exit 1' \
    --health-interval=2s \
    nginx:alpine
docker ps
# (health: starting)
sleep 2; docker inspect --format='{{.State.Health.Status}}' web
# healthy

# Instructing Docker Engine to relaunch container always or as main proc exits with error code; also Docker Engine will run the container after itself restart
docker-machine create --driver virtualbox sandbox
eval $(docker-machine env sandbox)
docker run -d -p 80:80 nginx:alpine
docker-machine restart sandbox
eval $(docker-machine env sandbox)
docker ps
# no container running
docker run -d -p 80:80 --restart always nginx:alpine
docker-machine restart sandbox
eval $(docker-machine env sandbox)
docker-machine ssh sandbox uptime
# now docker ps shows container still running
docker-machine rm sandbox

# reset with on-failure upon error exit code
docker run -d --restart=on-failure:5 alpine ash -c 'sleep 2 && /bin/false"
docker ps
docker ps
# 'Restarting (127)'
now=$(date "+%Y-%m-%dT%H:%M:%S")
docker events --until $now --filter 'container=boring_cray' | egrep 'die'
# Docker Engine increases the restart delay!! till hitting on-failure max num. of restarts

# Docker 1.12 enables 'daemonless' containers - stop/upgrade/restart Docker Engine WITHOUT affecting or restarting containers on the system!
docker-machine create sandbox --driver virtualbox --engine-opt live-restore
eval $(docker-machine env sandbox)
docker run -d -p 80:80 nginx:alpine
docker-machine ssh sandbox 
ps -ef | grep -E 'docker-|dockerd|nginx'
pgrep dockerd # 2653
sudo kill -9 2654
docker ps # Cannot connect to Docker daemon
pgrep nginx # 2813 2834
exit
curl $(docker-machine ip sandbox):80 # still works!!!

```



# K8s 

**POD a collection of containers and volumnes**

- single or multiple container(s) use case
- Example

YML manifest: 

```yaml
apiVersion: v1
kind: Pod
metadata:
	names: myapp-pod
	labels:
		app: myapp
spec:
	containers:

 - 	name: myapp-container
    	image: busybox
    	command: ['sh', '-c', 'echo Hello Kubernetes! && sleep 3600']
    	volumeMounts:
     -	mountPath: /test-pd
       	name: test-volume
       volumes:
    -	name: test-volume
      hostPath:
      	path: /data
      	type: Directory
```



**DEPLOYMENT describes desired state of SET of PODs and manages updates to them**

- Mostly define CONTROLER - mostly deployment (as opposed to a pod)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
	name: ngnix-deployment
	labels:
		app: nginx
spec:
	replicas: 3
	selector:
		matchLabels:
			app: nginx
	tempalte: 
		metadata:
			labels:
				app: nginx
		spec:
			containers:

   -	name: nginx
     			image: nginx:1.7.9
     			ports:
        -	containerPort: 80
```



**SERVICE routes traffic to SET-PODs**

- service-discovery smart routing
- various types: cluster-IP (contained cluster IP), node-port (node-specific port exposing), load-balancer (integrate with cloud provider ELB exposing to outside world)
- Internal routing (DNS-enabled linking PODs)
	<service-name>.<namespace>.cluter.local

```yaml
kind: Service
apiVersion: v1
metadata:
	name: my-service
sepc:
	selector:
		app: MyApp
	ports:
	-	name: http
		protocol: TCP
		port: 80
		targetPort: 9376
	- 	name: https
		portocol: TCP
		port: 443
		targetPort: 9377
```



**INGRESS expose a service to outside world**

- apart from load-balancer, rapidly costly ELB services with rise of pod service
- simply describes set of hosts and paths be routed to service from outside
- employ 'operator pattern' controler installed on cluster follwoing code specified

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
	annotations:
		kubernetes.io/ingress.class: ngnix
		kubernetes.io/ingress.provider: nginx
		kubernetes.io/tlk-acme: "true"
		ngnix.ingress.kubernetes.io/proxy-body-size: 512m
		ngnix.ingress.kubernetes.io/proxy-connect-timeout: "15"
		ngnix.ingress.kubernetes.io/proxy-read-timeout: "600"
	name: gitlab-unicorn
	namespace: gitlab
spec:
	rules:
	-	host: gitlab.moonswitch.io
		http:
			paths:
			-	backend:
					serviecName: gitlab-unicorn
					servicePort: 8181
				path: /
	tls:
	-	hosts:
		-	gitlab.moonswith.io
		secretName: gitlab-gitlab-tls
```



**JOB runs pod until completion**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
	name: api
sepc:
	template:
		spec:
			container:
			-	name: pi
				image: perl
				command: ["perl", "..."]
			restartPolicy: Never
	backoffLimit: 4
```



**HORIZONTAL POD AUTOSCALER out a deployment based observed CPU utilisation or other metrics**

- e.g. App aggrete cross-pod CPU 80%-90% -> busy -> scale up replicas, opposite also
	
	kubectl autoscale deployment php-apache --cpu-percent=50 --min=1 --max=10

- KEY in cluster-autoscaler in production
	- scale in node / resources!
	- elasticity of modern cloud - dynamic scaling based on defined metrics
