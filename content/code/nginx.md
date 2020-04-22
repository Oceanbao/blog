---
title: "Nginx"
date: 2020-04-17T23:15:22+08:00
showDate: true
draft: false
---


## Intro

> General description & explanation about nginx

## HTTP Proxying

> Capability to pass requests off to backend HTTP servers for further processing.

- Nginx is often set up as **reverse proxy solution to help scale out infras or to pass requests to other servers that are not designed to handle large client loads.
- Buffering and caching features for improving performance

> 1. Handle many concurrent connections at once; ideal for acting/proxying as Point-of-Contant for clients.
> 2. Fan-out to arbitrary backend servers, great for maintenance and scale-up-down.
> 3. Proxying achieved by manipulating a request aimed at Nginx server and passing it to other servers for actual processing; results passing back, relayed to client. The other servers (remote, local, virtual) aka **upstream servers****

### Basic HTTP Proxy Pass

- "proxy pass" handled by `proxy_pass` directive
- mainly found in location context; also valid in `if` blocks within location context and in `limit_except` contexts

```nginx
# server context

location /match/here {
    proxy_pass http://example.com;
}
```

- this config has no URI given at the end of server in the `proxy_pass` definition
- for definitions that fit this pattern, the URI requested by the client will be passed to the upstream server as-is
- e.g. when a request for `/match/here/please` is handled by this block, the request URI will be sent to `example.com` via HTTP as `http://example.com/match/here/please`



**Alternative Scenario**

```nginx
# server context

location /match/here {
    proxy_pass http://example.com/new/prefix;
}
```

- here the proxy server is defined with a URI segment on the end (/new/prefix) 
- when a URI is given in `proxy_pass` definition, the portion of the request matching the *location* definition is replaced by this URI during the pass
- e.g. a request for /match/here/please on Nginx server will be passed to the upstream server as `http://example.com/new/prefix/please` ; the /match/here is replaced by /new/prefix - **this is important point to keep in mind**

> sometimes this kind of replacement is impossible: then URI at end of `proxy_pass` is ignored and either the original URI from the client or the URI as modified by other directives will be passed to upstream server

- such as when location is matched using regex, Nginx cannot ascertain which part of URI matched, so it sends the original client request URI. Or when a rewrite directive is used within the same location, causing the client URI to be rewritten, but still handled in the same block - rewritten URI will be passed



### Header Processing

- **important** to pass **more than just the URI if expected the upstream server handle the request properly**. The request coming from Nginx on behalf of a client will look different than a request directly - as shown in **header**
- auto-adjust headers via proxying a request:
  - rid of any empty headers - rid of bloating
  - default treat any underscores invalid - removing these from proxied request - if wish as valid, set `underscores_in_headers` directive to "on"
  - "Host" header is re-written to the value defined by the `$proxy_host` - IP addr or name and port number of upstream, directly as defined by `proxy_pass`
  - "Connection" header changed to "close" - for signaling info about particular connection - here Nginx sets to "close" to indicate to upstream server that this conn will be closed once the original request is responded to - upstream shan't expect persistent conn
- effectively, any header **do not want to pass** should be set **empty string** 
- if backend app will be processing non-standard headers, **must ensure sans underscores** (or flags in http context or in the context of the default server declaration for IP/port combo)
- **Host header** is important in most proxying scenarios - default set to `$proxy_host`, a variable containing domain name or IP/port taken directly from `proxy_pass` definition - default being the only addr Nginx can be sure the upstream server responds to; **most common values for "Host" header**:
  - `$proxy_host` - "safe" from Nginx view, but often not needed by proxied server to correctly handle the request
  - `$http_host` - set to "Host" header from client request; headers sent by client are always available in Nginx as variables - they start with an `$http_prefix`, followed by header name in lowercase, with any dashes replaced by underscores - although `$http_host` works most times, when lcient request does not have valid "Host" header, this can cause pass to fail
  - `$host` - set in order of preference to: the host name from request line itself, the "Host" header from client request, or server name matching request
- In most cases, one would set "Host" header to `$host`, the most flexible and usually accurately filled in for the proxied servers



**Setting or Resetting Headers**

- to adjust headers, can use `proxy_set_header` directive - such as changing "Host" header as above, and adding some extra headers common with proxied requests:

```nginx
# server context

location /match/here {
    proxy_set_header HOST $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    
    proxy_pass http://example.com/new/prefix;
}
```

- `$host` should contain info of original host being requested
- `X-Forwarded-Proto` a general key (same in HAProxy) header gives the proxied server info on the schema of the original client request (http or https)
- `X-Real-IP` set to IP of client so that proxy can correctly make decisions or log based on this
- `X-Forwarded-For` header a list of IP of each server the client has been proxied through hitherto - e.g. above append Nginx server's IP to retrieved header from client



- Can add `proxy_set_header` directives out to server or http context, allowing it to be referenced in more than one location:

```nginx
# server context

proxy_set_header HOST $host;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

location /match/here {
    proxy_pass http://example.com/new/prefix;
}

location /different/match {
    proxy_pass http://example.com;
}
```



### Defining Upstream Context for LB Proxied Connections

- Above shows simple http proxy to single backend server. Here's scaling out to entire pools of backend servers
- One way is `upstream` directive to define a pool of servers - assuming any one of listed servers capable of handling client request - allowing scale out infra with almost no effort **must be set in http context** 

```nginx
# http context

upstream backend_hosts {
    server host1.example.com;
    server host2.example.com;
    server host3.example.com;
}    

server {
    listen 80;
    server_name example.com;
    
    location /proxy-me {
        proxy_pass http://backend_hosts;
    }
}
```

- `backend_hosts` name available as regular domain name used in `server`



### Changing Upstream Balancing Algorithm

- **round robin (default)** 
- **`least_conn`** : given to least number of active connections - useful in persistent backend
- **`ip_hash`** : IP-based routing - serving clients by same server each time - helpful in session consistency
- **`hash`** : mainly used in memcached proxying - servers divided based on value of arbitrary hash key - text, variable, or combo - the only algo requiring user provision of data

```nginx
# http context

upstream backend_hosts {
    
    least_conn;
    
    # hash $remote_addr$remote_port consistent;
    
    server host1.example.com;
    ...
}
```



### Setting Server Weight for Balancing

- default each server equally "weighted" to handle same amount of load

```nginx
# http context

upstream backend_hosts {
    server host1.example.com weight=3;
    ...
}
```

- host1 will receive 3x as other two servers (default weight =1)



### Using Buffers to Free Up Backend Servers

- One issue is impact of adding server to the process - most cases largely mitigated by taking advantage of Nginx's buffering and caching capabilities
- When proxying to another server, speed of two diff connections will affect client's experience:
  - conn from client to Nginx proxy
  - conn from Nginx proxy to backend server
- Nginx has ability to adjust its behaviour based on whichever one of these conn wish to optimise
- Without buffers, data is sent from **proxied server and immediately begins to be transmitted to client**. If clients assumed to be fast, buffering can be turned off for speed. With buffers, Nginx proxy will temp-store backend's response and then feed data to the client - if client slow, allows Nginx server to close conn to backend sooner - can then handle distributing the data to client at whatever pace possible
- Default buffering design since clients tend to have vastly diff speeds - can adjust with directives (set in http, server, or location contexts) **important: sizing directives are configured per request, so increasing them beyond need can affect speed when many client requests** 
  - **`proxy_buffering`** :  enable context and child contexts buffering, default 'on'
  - **`proxy_buffers`**: number (first arg) and size (second arg) of buffers for proxied responses - default to 8 buffers of size == one memory page (4k or 8k)
  - **`proxy_buffer_size`** : initial portion of response from backend server containing headers is buffered separately from rest of response - set size of buffer for this portion - default same size == above
  - **`proxy_busy_buffers_size`** : set max size of buffers can be marked "client-ready" and thus busy - while client can only read data from one buffer a time, buffers are placed in queue to send client in bunches - controls size of buffer space allowed to be in this state
  - **`proxy_max_temp_file_size`** : max size, per request, for temp-file on disk - created when upstream response is too large to fit into a buffer
  - **`proxy_temp_file_write_size`** : amount of data Nginx will write to the temp-file at one time when proxied server's response is too large for configured buffers
  - **`proxy_temp_path`**: path to area on disk where Nginx should store any temp-fiels when response from upstream server cannot fit into configured buffers
- probably most useful directive to adjust are `proxy_buffers` and `proxy_buffer_size`
- Example: up number of available proxy buffers for each upstream request, while trimming down buffer that likely stores the headers

```nginx
# server context

proxy_buffering on;
proxy_buffer_size 1k;
proxy_buffers 24 4k;
proxy_busy_buffers_size 8k;
proxy_max_temp_file_size 2048m;
proxy_temp_file_write_size 32k;

location / {
    proxy_pass http://example.com;
}
```

- In contrast, if having fast client able to take on speed conn, can turn buffering off completely - actually still use buffers if upstream is faster than client, but will immediately try to flush dat to client instead of waiting for buffer to pool

```nginx
# server context

proxy_buffering off;
proxy_buffer_size 4k;

...
```



### High Availability

- Making robust via redundant set of LBs

![Image](https://assets.digitalocean.com/articles/high_availability/ha-diagram-animated.gif)



### Caching to Reduce Response Times

- While buffering can help free up backend server to handle more requests, Nginx also provides way to cache content from backend servers, eliminating need to connect to upstream at all for many requests
- `proxy_cache_path` directive to create an area where data returned from proxied servers be kept (**must set in http context**)

```nginx
# http context

proxy_cache_path /var/lib/nginx/cache levels=1:2 keys_zone=backcache:8m max_size=50m;
proxy_cache_key "$scheme$request_method$host$request_uri$is_args$args";
proxy_cache_valid 200 302 10m;
proxy_cache_valid 404 1m;
```

- If cache area not exist, can create it with correct permission and ownership

```shell
sudo mkdir -p /var/lib/nginx/cache
sudo chown www-data /var/lib/nginx/cahce
sudo chmod 700 /var/lib/nginx/cache
```

- `levels=` specifies how cache will be organised - a cache key created by hashing the value of a key (configured below) - "a single char dir (last char of hashed value) with two chars subdir (taken from next two chars from end of hased value)"
- `keys_zone` defines name for cache zone, also how much metadata to store is defined here. e.g. 8MB of keys each MB can store 8000 entries - `max_size` sets max size of actual cached data
- `proxy_cache_key` set the key used to store cached values - same key used to check if a request can be served from cache - e.g. setting this as combo of scheme (http/https), the HTTP request method, and requested host and URI
- `proxy_cache_valid` can be specified multiple times to configure how long to store values depending on status code - e.g. store successes and redirects for 10 minutes expiring cache for 404 responses every minute
- Below tells Nginx when to use cache - in locations where proxy to a backend

```nginx
# server context

location /proxy-me {
    proxy_cache backcache;
    proxy_cache_bypass $http_cache_control;
    add_header X-Proxy-Cache $upstream_cache_status;
    
    proxy_pass http://backend;
}
```

- `bypass` set contains indicator as to whether client is explicitly requesting a fresh, non-cached version of resource - allowing Nginx to correctly handle these types 
- `X-Proxy-Cache` basically sets header to see if request resulted in a cache hit, miss or explicitly bypassed - esp valuable for debugging

**Notes**

- Beware of any user-related data should NOT be cached - erroneously presenting other users data
- For private content, set `Cache-Control` header to "no-cache", "no-store", or "private"
  - `no-cache` used if data is dynamic and important - ETag hashed metadata header is checked on each request and previous value can be served if backend returns the same hash value
  - `no-store` at no point should data received ever be cached - safest for private data
  - `private` no shared cache space should cache this data - useful for indicating user's browser can cache data but proxy server shouldn't consider this dat valid for subsequent requests
  - `public` can be cached at any point in connection
  - `max-age` controls number of seconds any resource should be cached
- If backend also uses Nginx, can set some of this using `expires` which will set `max-age` for `Cache-Control`:

```nginx
location / {
    expires 60m;
}

location /check-me {
    expires -1;
}
```

- first block allows content be cached for an hour, second sets "no-cache"; to set other values, use `add_header` like so:

```nginx
location /private {
    expires -1;
    add_header Cache-Control "no-store";
}
```



## HAProxy to LB MySQL

https://www.digitalocean.com/community/tutorials/how-to-use-haproxy-to-set-up-http-load-balancing-on-an-ubuntu-vps



https://www.digitalocean.com/community/tutorials/how-to-use-haproxy-to-set-up-mysql-load-balancing--3

