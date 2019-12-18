---
title: "ZeroMQ"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

# Demo: Python-Flask-ZMQ-Docker Stack

## 2-Way-2-Procs Communication Microservices

**ZeroMQ** supplies basis for developing scalable distribution systed on top of socket. It is secure - ellipic curve cryptography (V4) and out-of-box communication patterns.

[ZeroMQ Messaging](https://www.digitalocean.com/community/tutorials/how-to-work-with-the-zeromq-messaging-library)

[http://blog.pythonisito.com/2012/08/distributed-systems-with-zeromq.html](http://blog.pythonisito.com/2012/08/distributed-systems-with-zeromq.html)

Starting with simple **PUB-SUB** and Flask inside docker.

![Image](https://blog.apcelent.com/images/microservices-docker-python-apcelent.png)

- 1 & 3: Flask server running on port-5000 with endpoint. URL caters to GET req, and all req of the format `?Params=` has response where upper case letters will be converted to lower case before returned
- 2: Response message also sent to ZMQ Publisher running in the same container
- 4 & 5: ZMQ Subscriber keeps listening and saves message from ZMQ Server to a file called subscriber.log

**Server**

```dockerfile
FROM ubuntu:14.04

RUN apt-get update \
		apt-get install -y --force-yes python python-dev python-setuptools software-properties-common gcc python-pip \
		apt-get clean all \
		pip install pyzmq Flask

ADD zmqserver.py /tmp/zmqserver.py

# Flask Port
EXPOSE 5000

# Zmq Sub Server
EXPOSE 4444

CMD ["python","/tmp/zmqserver.py"]
```

**zmqserver.py**

```python
# server.py
import time
import zmq

HOST = '127.0.0.1'
PORT = '4444'

_context = zmq.Context()
_publisher = _context.socket(zmq.PUB)
url = 'tcp://{}:{}'.format(HOST, PORT)


def publish_message(message):
  # url = “tcp://192.168.10.10:5555”

  try:
    _publisher.bind(url)
    time.sleep(1)
    print(f'Sending message: {message, _publisher}')
    _publisher.send(message)

  except Exception as e:
    print("error {}".format(e))

  finally:
    # To unbind publisher to keep receiving published messages
    # Or else "Address already in use Error"
    _publisher.unbind(url)


from flask import Flask
from flask import request
app = Flask(__name__)

# Endpoint for printing and publishing 
@app.route("/downcase/", methods=['GET'])
def lowerString():

  _strn = request.args.get('param')
  response = b'lower case of {} is {}'.format(_strn, _strn.lower())
  publish_message(response)
  return response

if __name__ == '__main__':
  # default port running at 5000
  app.run(host='0.0.0.0', debug=False)
```



**Build Publisher and Run**

```bash
sudo docker build -t zmq-pub .
docker run --name pub-server -p 5000:5000 -p 4444:4444 -t zmq-pub
```



**zmqclient.py**

```python
# client.py
import zmq
import sys
import time
import logging
import os

HOST = '127.0.0.1'
PORT = '4444'

logging.basicConfig(filename='subscriber.log', level=logging.INFO)


class ZClient(object):

    def __init__(self, host=HOST, port=PORT):
    """Initialize Worker"""
    self.host = host
    self.port = port
    self._context = zmq.Context()
    self._subscriber = self._context.socket(zmq.SUB)
    print("Client Initiated")

    def receive_message(self):
    """Start receiving messages"""
    # “tcp://192.168.10.10:5555″
    self._subscriber.connect('tcp://{}:{}'.format(self.host, self.port))
    self._subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    while True:
        print('listening on tcp://{}:{}'.format(self.host, self.port))
        message = self._subscriber.recv()
        print(message)
        logging.info(
            '{}   - {}'.format(message, time.strftime("%Y-%m-%d %H:%M")))

if __name__ == '__main__':
    zs = ZClient()
    zs.receive_message()
```



**Run Client**

`python zmqclient.py`

Requesting at `localhost:5000/downcase/?Param=<String with mixed case letters>`

> Messages from publisher will be sent over to subscriber.
>
> Additonally, it logs to a file called subscriber.log



## Async Client/Server in Python

```python
# async_zmq.py

import zmq
import sys
import threading
import time
from random import randint, random

__author__ = "Felipe Cruz <felipecruz@loogica.net>"
__license__ = "MIT/X11"

def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

class ClientTask(threading.Thread):
    """ClientTask"""
    def __init__(self, id):
        self.id = id
        threading.Thread.__init__ (self)

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'worker-%d' % self.id
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:5570')
        print('Client %s started' % (identity))
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)
        reqs = 0
        while True:
            reqs = reqs + 1
            print('Req #%d sent..' % (reqs))
            socket.send_string(u'request #%d' % (reqs))
            for i in range(5):
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    msg = socket.recv()
                    tprint('Client %s received: %s' % (identity, msg))

        socket.close()
        context.term()

class ServerTask(threading.Thread):
    """ServerTask"""
    def __init__(self):
        threading.Thread.__init__ (self)

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5570')

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        workers = []
        for i in range(5):
            worker = ServerWorker(context)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class ServerWorker(threading.Thread):
    """ServerWorker"""
    def __init__(self, context):
        threading.Thread.__init__ (self)
        self.context = context

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        tprint('Worker started')
        while True:
            ident, msg = worker.recv_multipart()
            tprint('Worker received %s from %s' % (msg, ident))
            replies = randint(0,4)
            for i in range(replies):
                time.sleep(1. / (randint(1,10)))
                worker.send_multipart([ident, msg])

        worker.close()

def main():
    """main function"""
    server = ServerTask()
    server.start()
    for i in range(3):
        client = ClientTask(i)
        client.start()

    server.join()

if __name__ == "__main__":
    main()
```



**`python async_zmq.py`**

```bash
Worker started
Client worker-0 started
Worker started
Worker started
Client worker-2 started
Req #1 sent..
Worker started
Req #1 sent..
Worker started
Client worker-1 started
Worker received b'request #1' from b'worker-2'
Worker received b'request #1' from b'worker-0'
Req #1 sent..
Worker received b'request #1' from b'worker-1'
Client worker-1 received: b'request #1'
Client worker-2 received: b'request #1'
Client worker-2 received: b'request #1'
Client worker-0 received: b'request #1'
Client worker-0 received: b'request #1'
Req #2 sent..
Worker received b'request #2' from b'worker-2'
Req #2 sent..
Worker received b'request #2' from b'worker-0'
Client worker-0 received: b'request #2'
Client worker-2 received: b'request #2'
Client worker-0 received: b'request #2'
Client worker-2 received: b'request #2'
Client worker-0 received: b'request #2'
Req #2 sent..
Worker received b'request #2' from b'worker-1'
Client worker-0 received: b'request #2'
Req #3 sent..
Worker received b'request #3' from b'worker-0'
Client worker-0 received: b'request #3'
Client worker-0 received: b'request #3'
Client worker-0 received: b'request #3'
Client worker-0 received: b'request #3'
Req #3 sent..
Worker received b'request #3' from b'worker-2'
Req #4 sent..
Worker received b'request #4' from b'worker-0'
Client worker-2 received: b'request #3'
Client worker-2 received: b'request #3'
Client worker-2 received: b'request #3'
Req #3 sent..
Worker received b'request #3' from b'worker-1'
Client worker-1 received: b'request #3'
Client worker-1 received: b'request #3'
Req #4 sent..
Worker received b'request #4' from b'worker-2'
Client worker-1 received: b'request #3'
Req #4 sent..
Worker received b'request #4' from b'worker-1'
Client worker-1 received: b'request #4'
Req #5 sent..
...
...
```

