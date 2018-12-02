---
title: "Scrapy"
date: 2018-10-30T17:43:17-04:00
showDate: true
draft: false
---

<h1>Learning Scrapy</h1>


## Table of Contents
1. [Basic Crawling](#basic)
2. [Mobile App](#mobile)
3. [Spider Recipes](#recipe)
4. [Scrapinghub](#hub)
5. [Configuration & Management](#config)
6. [Programming Scrapy](#programming)
7. [Pipeline Recipe](#pipeline)
8. [Official Tutorial](#official)



# Basic Crawling 
<a name="basic"></a> 

### $UR^2IM$

- URL
- Request
- Response
- Items
- More URLs (recurring to Request)

`scrapy shell -s USER_AGENT="Mozilla/5.0" <URL>`

`response.body[:50]`

Actual value is gained via `extract()` or `re()`



### Scrapy Project

Shell is mere utility aiding testing, real codes start with Project.

`scrapy startproject properties`

This chapter focuses on `items.py` and `spiders` directory. 

### Defining ITEMS

- Redefine class to fitting name
- NOTE: Declaring a field NOT equal filling it on every spider
- Fields
  - **images** - images pipeline will auto-fill this based on `image_urls` 
  - **location** - Geocoding pipeline will auto-fill this
  - Self-defined *housekeeping* fields for debugging
    - url - response.url
    - project - self.settings.get('BOT_NAME')
    - spider - self.name
    - server - socket.gethostname()
    - date - datetime.datetime.now()

- With a list of fields, it's easy to mod and cutomise `class` default:

```python
from scrapy.item import Item, Field

class PropertiesItem(Item):
    # Primary fields
    title = Field()
    price = Field()
    description = Field()
    address = Field()
    image_urls = Field()
    
    # Calculated fields
    images = Field()
    location = Field()
    
    # Housekeeping fields
    url = Field()
    project = Field()
    spider = Field()
    server = Field()
    date = Field()
```



### Writing Spiders

Halfway, typically one spider per website or a section of website if large. A spider code implements $UR^2IM$ process. TIP: spider or project? A project groups `items` and spiders, designed for same type over many sites, as above can be used generally.

`scrape genspider basic/crawl web`

TIP: Scrapy has many subdir but all cmd assumes root dir where lies `scrapy.cfg` file. Whenever referring to 'packages and modules', they are set as to map to directory structure. E.g. `ocean.spiders.basic` is under `ocean/spiders` directory.

The `self` reference in `parse()` enables functionality of spider.

Start coding and use `log()` to output info in the primary fields table.

```python
def parse(self, response):
    self.log("title: %s" % response.xpath('//*[@itemprop="name"][1]/text()').extract())
    # similarly for others
```

`scrapy crawl`

> `self.log()` output DEBUG: sessions for inspecting correctness

`scrapy parse` 

This allows to use **most suitable** spider to parse any URL as ARG. BUT best specify.

`scrapy parse --spider=crawl <URL>`

This outputs similar info as above and often used for DEBUGGING.



### Populating ITEM

Slight mod yet "unlocking" tons of functionalities. 

INIT and return one. Adding to `parse()` function process.

```python
from properties.items import PropertiesItem

# inside parse()
item = PropertiesItem()
item['title'] = response.xpath('//*[@itemprop="name"][1]/text()').extract()
# et les restes 
return item
```

<a name="pipeline"></a>

Now `scrape crawl basic` returns not LOG but DICT of the item. Scrapy is built around the ITEMs to be used by PIPELINEs for more functionalities.

### Saving to Files

`scrape crawl basic -o items.json .jl .csv .xml`

CSV and XML popular for excel apps. JSON for expressiveness and link to JavaScript. `.jl` files have one JSON object per line, read more efficiently.

To save on cloud:

`scrape crawl basic -o "ftp://user:pass@ftp.scrapybook.com/item-s.json"`

`scrapy crawl basic -o "s3://aws_key:aws_secret@scrapy-book/items.json"`

Scrapy parse now adjusted to the new setting. You'll appreciate it even more while DEBUG URLs that give unexpected results.

### Clean Up - ITEM LOADER and housekeeping fields

`ItemLoader` class replaces all messy looking `extract()` and `xpath()` operations.

```python
from scrapy.loader import ItemLoader

def parse(self, response):
    l = ItemLoader(item = PropertiesItem(), response = response)
    l.add_xpath('title', '//*[@itemprop="name"][1]/text()')
    l.add_xpath('price', './/[@itemprop="price"][1]/text()', re = '[,.0-9]+')
    return l.load_item()
```

More than clean, it declares very clearly intention of action. `ItemLoader` provie many cool ways of mixing data, formatting, cleaning up. Note they are actively developed so keep abreast [here](http://doc.scrapy.org/en/latest/topics/loaders.html)  

**Processors** are fast and neat functions manipulating multiple selectors. 

| Processor                                                 | Functionality                                                |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| `Join()`                                                  | Concatenates multiple results into one                       |
| `MapCompose(unicode.strip)`                               | Chaining python func: i.e. Removes leading and trailling whitespace chars |
| `MapCompose(unicode.title)`                               | Also gives title cased results                               |
| `MapCompose(float)`                                       | converts strings to integers                                 |
| `MapCompose(lambda i: i.replace(',', ''), float)`         | WOW, inheriting all power of lambda function....             |
| `MapCompose(lambda i: urlparse.urljoin(response.url, i))` | converts relative URLs to absolute using `response.url` as base |

Possible use of any Python expression as a processor. These are simply functions embedded in Scrapy, such that possible to try in SHELL

```shell
# scrapy shell someweb
from scrapy.loader.processors import MapCompose, Join

Join()(['hi', 'John'])
>>> u'hi John'
```

Let's see how to add them inside `parse()`

```python
# RECALL to import relevant modules
import datetime, socket, urlparse

# for processing items
l.add_xpath('title', 'XPATH', MapCompose())

# for easy, housekeeping fields
l.add_value('url', response.url)
l.add_value('project', self.settings.get("BOT_NAME"))
l.add_value('spider', self.name)
l.add_value('server', socket.gethosename())
l.add_value('date', datetime.datetime.now())
```

> Perfectly looking `Items` and might at first glance seems complex. BUT it's worth it, especially considering the similar power requires tons more codes in other langues, here only 25-line of codes.
>
> Another feeling stems from all those processors and `ItemLoaders` These xeno-python codes are worth the effort for serious web scraping journey.

### Creating Contracts

Contracts like unit tests for spiders, for quickly checking broke code. For instance, checking old spiders if working now. Contracts are included in the comments just after the name of function (docstring) starting with `@` 

```python
def parse(self, response):
    """ This function parses a property page.
    
    @url http://web:9312/proeprties/property_000000.html
    @returns items 1
    @scrapes title price description address image_urls
    @scrapes url project spider server date
    """
```

It means 'checking this URL and you should fine one item with values on those fields enlisted here'. 

`scrapy check`  will go and check whether the contracts are valid : `scrapy check basic` In case of error

```shell
FAIL: [basic] parse (@scrapes post-hook)
------------------------------------------------------------------------------------
ContractFail: 'url' field is missing
```

Fail either in code or selector. Good first line of check.

#### RECAP CODE

```python
import datetime
import urlparse
import socket
import scrapy

from scrapy.loader.processors import MapCompose, Join
from scrapy.loader import ItemLoader
from scrapy.http import Request

from properties.items import PropertiesItem


class BasicSpider(scrapy.Spider):
    name = "manual"
    allowed_domains = ["web"]

    # Start on the first index page
    start_urls = (
        'http://web:9312/properties/index_00000.html',
    )

    def parse(self, response):
        # Get the next index URLs and yield Requests
        next_selector = response.xpath('//*[contains(@class,"next")]//@href')
        for url in next_selector.extract():
            yield Request(urlparse.urljoin(response.url, url))

        # Get item URLs and yield Requests
        item_selector = response.xpath('//*[@itemprop="url"]/@href')
        for url in item_selector.extract():
            yield Request(urlparse.urljoin(response.url, url),
                          callback=self.parse_item)

    def parse_item(self, response):
        """ This function parses a property page.

        @url http://web:9312/properties/property_000000.html
        @returns items 1
        @scrapes title price description address image_urls
        @scrapes url project spider server date
        """

        # Create the loader using the response
        l = ItemLoader(item=PropertiesItem(), response=response)

        # Load fields using XPath expressions
        l.add_xpath('title', '//*[@itemprop="name"][1]/text()',
                    MapCompose(unicode.strip, unicode.title))
        l.add_xpath('price', './/*[@itemprop="price"][1]/text()',
                    MapCompose(lambda i: i.replace(',', ''), float),
                    re='[,.0-9]+')
        l.add_xpath('description', '//*[@itemprop="description"][1]/text()',
                    MapCompose(unicode.strip), Join())
        l.add_xpath('address',
                    '//*[@itemtype="http://schema.org/Place"][1]/text()',
                    MapCompose(unicode.strip))
        l.add_xpath('image_urls', '//*[@itemprop="image"][1]/@src',
                    MapCompose(lambda i: urlparse.urljoin(response.url, i)))

        # Housekeeping fields
        l.add_value('url', response.url)
        l.add_value('project', self.settings.get('BOT_NAME'))
        l.add_value('spider', self.name)
        l.add_value('server', socket.gethostname())
        l.add_value('date', datetime.datetime.now())

        return l.load_item()
```

**With CrawlSpider**

```python
import datetime
import urlparse
import socket

from scrapy.loader.processors import MapCompose, Join
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.loader import ItemLoader

from properties.items import PropertiesItem


class EasySpider(CrawlSpider):
    name = 'crawl'
    allowed_domains = ["web"]

    # Start on the first index page
    start_urls = (
        'http://web:9312/properties/index_00000.html',
    )

    # Rules for horizontal and vertical crawling
    rules = (
        Rule(LinkExtractor(restrict_xpaths='//*[contains(@class,"next")]')),
        Rule(LinkExtractor(restrict_xpaths='//*[@itemprop="url"]'),
             callback='parse_item')
    )

    def parse_item(self, response):
        """ This function parses a property page.

        @url http://web:9312/properties/property_000000.html
        @returns items 1
        @scrapes title price description address image_urls
        @scrapes url project spider server date
        """

        # Create the loader using the response
        l = ItemLoader(item=PropertiesItem(), response=response)

        # Load fields using XPath expressions
        l.add_xpath('title', '//*[@itemprop="name"][1]/text()',
                    MapCompose(unicode.strip, unicode.title))
        l.add_xpath('price', './/*[@itemprop="price"][1]/text()',
                    MapCompose(lambda i: i.replace(',', ''), float),
                    re='[,.0-9]+')
        l.add_xpath('description', '//*[@itemprop="description"][1]/text()',
                    MapCompose(unicode.strip), Join())
        l.add_xpath('address',
                    '//*[@itemtype="http://schema.org/Place"][1]/text()',
                    MapCompose(unicode.strip))
        l.add_xpath('image_urls', '//*[@itemprop="image"][1]/@src',
                    MapCompose(lambda i: urlparse.urljoin(response.url, i)))

        # Housekeeping fields
        l.add_value('url', response.url)
        l.add_value('project', self.settings.get('BOT_NAME'))
        l.add_value('spider', self.name)
        l.add_value('server', socket.gethostname())
        l.add_value('date', datetime.datetime.now())

        return l.load_item()

```



### MORE URLS

First kind, hardcode LIST of URLs in `start_urls = ()`

Up a notch would be `start_urls = [i.strip() for i in open('todo.urls.txt').readlines()]`

**Crawling Direction** 

- Horizontal - from index to another
  - Find **Next Page** icon, then parse `//*[contains(@class, "next")]//@href` 
- Vertical - from index page to the listing pages to extract `Items`
  - Find **Listing** page URL via similar method

*Check via Shell first*

### Two-Direction Crawling

```python
from scrapy.http import Request
def parse(self, response):
    # Get next index and yield Requests
    next_selector = response.xpath('//@[contains(@class, 'next')]//@href')
    
    for url in next_selector.extract():
        yield Request(urlparse.urljoin(response.url, url))
    
    # Get item URLs and yield Requests
    item_selector = response.xpath('//*[itemprop='url']/@href')
    
    for url in item_selector.extract():
        yield Requests(urlparse.urljoin(response.url, url), 
                      callback = self.parse_item)
```

> `yield` DOES NOT EXIT function, but continues with the `for` loop. PYTHON MAGIC.

For testing purpose, stop at certain items quantity

`scrapy crawl manual -s CLOSESPIDER_ITEMCOUNT=90`

> It first read index, then spawns many Requests, executed. Scrapy uses LIFO strategy to process requests (depth first crawl). Last request submitted will be processed first. Convenient for most cases. E.g. processing each listing page before moving to the next index page, or else fill a huge queue of pending listing pages. 
>
> Modifiable in setting PRIORITY argument greater than 0 (higher than default) or less than 0. In general, scrapy scheduler will execute higher priority requests first, but don't spend much time thinking about the exact request should be executed first. Highly likely that not use more than one or two request priority levels in most applicaitons. Note also that URLs are subject to duplication filtering, most often desired. If wishing to perform a request to the same URL more than once, `dont_filter = true` inside `Request()`

### Two-Direction Crawling with CrawlSpider

Seemingly tedious code in basic spider, CrawlSpider class offers simpler methods.  Once genspider, the extra code inherited on the surface are:

```python
rules = ( 
	Rule(LinkExtractor(allow=r'Items/'), callback = 'parse_item', follow = true)
)

def parse_item(self, resposne):
    pass
```

TIP Why learn manual as above? `yield` + `Requests` with `callback` is such as USEFUL and CORE technique that will use repeated later, worth knowing.

Now mode rules one for horizontal and one for vertical crawlling

```python
rules = (
	Rule(LinkExtractor(restrict_xpaths = '//*[contains(@class, 'next')]')),
    Rule(LinkExtractor(restrict_xpaths = '//*[@itemprop='url']'), callback = 'parse_item')
)
```

>  What differ are missing `a` and `href` constraints, for LinkExtractor by default looks for those two `elemetns`. Also note taht callbacks are now strings not method references in `Requests(self.parse_item)`. Unless `callback` is set, a `Rule` will follow the extracted URLs, which means that it will scan target pages for extra links and follow them. If a `callback`, `Rule` will NOT follow the links from target pages. If need to follow links, either `return/yield` them from `callback` method, or set `follow=true` in `Rule()`. This might be useful when listing pages contain both ITEMs and extra useful navigation links!!



# From Scrapy to a MOBILE APP

### Choosing a Mobile App Framework

Feeding data scraped to app is easy if using appropriate tools. Many frameworks such as PhoneGap, Appcelerator, jQuery Mobile, Sencha Touch.

This tutorial uses Appery.io for its iOS, Android, Windows Phone and HTML5 compatibility and ease of use using PhoneGap and jQuery Mobile. Its paid service bundles both mobile and backend services, meaning no need to configure DB, write REST APIs or use perhaps other langues to write them. 

Detail see **Learning Scrapy** on GitHub source code.

# Quick Spider Recipes

Previously on extracting info from pages and stored as Items. This is the 80% of the use case, and this section covers special usage to focus on 2 two important classes `Request` and `Response`, the two Rs in the process model.

### A spider that logs in

When website having login mechanism, the two Rs are key to extract data via inspecting Network traffic in dev tool.

Once login correctly, **Request Method: POST** appears in network request. Inspect data including **Form Data**, **Cookie** stores the login detail set under **Request Headers**. Thus a single operation, such as login, may involve several server round-trips, including POST, HTTP redirects 302, etc. Scrapy handles most of these operations automatically, with simple code needed. Inherit from CrawlSpider, define a new spider: ```class LoginSpider(CrawlSpider): name='login'`

`FormRequest` class send initial request that logs in by performing POST request, similar to `Request` with extra `formadata` argument to pass data (i.e. `user` and `pass`)

```python
import datetime
import urlparse
import socket

from scrapy.loader.processors import MapCompose, Join
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest

from properties.items import PropertiesItem


class LoginSpider(CrawlSpider):
    name = 'login'
    allowed_domains = ["web"]

    # Start with a login request
    def start_requests(self):
        return [
            FormRequest(
                "http://web:9312/dynamic/login",
                formdata={"user": "user", "pass": "pass"}
            )]

    # Rules for horizontal and vertical crawling
    rules = (
        Rule(LinkExtractor(restrict_xpaths='//*[contains(@class,"next")]')),
        Rule(LinkExtractor(restrict_xpaths='//*[@itemprop="url"]'),
             callback='parse_item')
    )

    def parse_item(self, response):
        # stay the same
```

That's it really. The default `parse()` of `CrawlSpider` handles `Response` and uses `Rules` exactly as previously. So little code since Scrapy handles cookies transparently for us, as soon as login, it passes them on to subsequent requests in exactly the same manner as a browser.

Naturally some login mechanism is more complex, such as a **HIDDEN** value, which need be POST together. This means two requests! Visit the form page and then the login page, then pass through some data. A new spider now in `start_requests()` return a simple `Request` to our form page, and will manually handle the ersponse by setting its `callback` to hour handler method named `parse_welcome()` below. In it, use the helper `from_response()` method of `FormRequest` object to create `FormRequest` that is pre-populated with all the fields and values from the original form. `FormRequest.from_response()` roughly emulates a submit click on the first form on the page with all the fields left blank.

TIP: worth familiarise with documentaion of `from_response()` for its many features like `formname` and `formnumber` designed to help select the form desired if multiple occur. 

This effortless feature use `formdata` argument to fill in the user and pass fields and return `FormRequest`:

```python
import datetime
import urlparse
import socket

from scrapy.loader.processors import MapCompose, Join
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.loader import ItemLoader
from scrapy.http import Request, FormRequest

from properties.items import PropertiesItem


class NonceLoginSpider(CrawlSpider):
    name = 'noncelogin'
    allowed_domains = ["web"]

    # Start on the welcome page
    def start_requests(self):
        return [
            Request(
                "http://web:9312/dynamic/nonce",
                callback=self.parse_welcome)
        ]

    # Post welcome page's first form with the given user/pass
    def parse_welcome(self, response):
        return FormRequest.from_response(
            response,
            formdata={"user": "user", "pass": "pass"}
        )
```

When run in Shell, observes first GET to /hidden_login page, and then POST, folllowed by redirection on to /hidden_login_success page that leads to /gated as before.

### A spider that uses JSON APIs and AJAX pages

Hidden elements managed by JSON objects dynamically. Similarly, inspect via Network traffic, often in the form of **api.json**. More complex APIs may require login, POST, or return more interesting data structures. At any rate, JSON is one of the easiest formats to parse as no need to write any XPATH to extract.

Python provides a great JSON parsing module - `import json` -> `json.loads(response.body)` to parse JSON and convert it to an equal object consisting of Python primitives, lists, and dicts.

Once found, make spider that works just on it.

`start_urls = ('http://someurl/api.json')`

More complex need can be done using previous mechanism. At this point, Scrapy will open this URL and call `parse()` with `Response` as argument. 

```python
import urlparse
import socket
import scrapy
import json

class ApiSpider(scrapy.Spider):
    name = 'api'
    allowed_domains = ["web"]

    # Start on the first index page
    start_urls = (
        'http://web:9312/properties/api.json',
    )

    # Format the URLs based on the API call response
    def parse(self, response):
        base_url = "http://web:9312/properties/"
        js = json.loads(response.body)
        for item in js:
            id = item["id"]
            title = item["title"]
            url = base_url + "property_%06d.html" % id
            yield Request(url, meta={"title": title}, callback=self.parse_item)

```

`%06d` is a very useful piece of Python syntax for creating new strings by combining Python variables. `%d` means treat as digit and extends to 6 characters by prepending 0s if necessary. If `id` has the value 5, it will be repalced with 000005, else if 34322 then 034322. `yield` new `Request` of correctly joined URL with callback.

### Passing arguments between responses

If info on JSON APIs need be stored to ITEM, how to pass it from `parse()` to `parse_item()` method?

**meta** data as dict inside `Request()` used for this purpose, and the index page info. For example, let's set a title value on this dict to store the title from JSON object: `title = item["title"]`

`yield Request(url, meta = {"title": title}, callback = self.parse_item)`

Inside `parse_item()`, we can use this value instead of XPath expression before:

`l.add_value('title', response.meta['title'], MapCompose(unicode.strip, unicode.title))`

Notice the switch from calling `add_xpath()` to `add_value()` when using **meta** 

### 30x Faster Spider

Avoid scraping every single listing page if able to extract about the same info from index page !!

TIP: If a website gives 10, 50 or 100 listing pages per index page by tuning a param, such as `&show=50` on URL, set to maximum before horizontal crawling.

A programming design decision here, since most website set **throttle** requests.

Demo of such mechanism:

```python
def parse(self, response):
    # Get next index URLs and yield Requests
    # same as before
    
    # Iterate through products and create PropertiesItems
    selectors = response.xpath(
    	'//*[@itemtype = "http://scheme.org/Product"]')
    # differs in yielding each of 30 product from selectors and parse_item them
    for selector in selectors:
        yield self.parse_item(selector, response)

def parse_item(self, selector, response):
    # Create the laoder using the selector
    l = ItemLoader(item = PropertiesItem(), selector = selector)
    
    # Load fields using XPath
    # NOTE!! relative CURRENT XPath selector '.' for each expression!
    l.add_xpath('title', './/*[@itemprop="name"][1]/text()',
    	MapCompose(unicode.strip, unicode.title))
    # etc
    make_url = lambda i : urlparse.urljoin(response.url, i)
    l.add_xpath('image_urls', './/*[@itemprop="image"][1]/@src',
               MapCompose(make_url))
    # Housekeeping mostly the same
    l.add_xpath('image_urls', './/*[itemprop="url"][1]/@href',
               MapCompose(make_url))
    
    return l.load_item()
```

Slight changes made:

- ItemLoader now uses selector as source rather than Response. This is a convenient feature of `ItemLoader` API allowing to extract from currently selected segment instead of entire page.
- Path turned to relative by prepending dot (.) TIP: so happens in this case, XPath identical in detail and index pages, but not always the case
- Need to compile URL of ITEM manually. Before `response.url` was giving URL for listing page, now gives URL of index page since this was page crawled. Need to extract URL of listing using XPath and convert it to absolute URL with MapCompose processor

### Spider crawling based on Excel file

In case where scrape data from many sites with ONLY XPath changes, overkill to have a spider for every site. How to use a single spider?

Create a new project `generic` name spider `fromcsv`. Create a CSV with fields containing relevant URL, items to extract (XPATH), save in project root directory.

Read the CSV into Dict:

```python
import csv

with open("data.csv", "rU") as f:
    reader = csv.DictReader(f)
    for line in reader:
        print(line)
```

Modification to spider:

- remove `start_urls` and `allows_domains`
- use `start_requests()` and `Request` each row of data
- Store field names and XPATH from CSV in `request.meta` to use in `parse()` 

```python
import csv

import scrapy
from scrapy.http import Request
from scrapy.loader import ItemLoader
from scrapy.item import Item, Field


class FromcsvSpider(scrapy.Spider):
    name = "fromcsv"

    def start_requests(self):
        with open(getattr(self, "file", "todo.csv"), "rU") as f:
            reader = csv.DictReader(f)
            for line in reader:
                request = Request(line.pop('url'))
                request.meta['fields'] = line
                yield request

    def parse(self, response):
        item = Item()
        l = ItemLoader(item=item, response=response)
        for name, xpath in response.meta['fields'].iteritems():
            if xpath:
                item.fields[name] = Field()
                l.add_xpath(name, xpath)

        return l.load_item()
```

Observation:

- Since no project-wide ITEM, need to provide one to ItemLoader manually inside `parse()`
- Fields added dynamically using `fields` member variable of Item.
- Hardcoding data.csv is not good practice, Scrapy gives easy way to pass arguments to spiders.
  - `-a variable=value` 
  - a spider property is set and able to retrieve it with `self.variable` 
  - to check for variable and use a default if it isn't provided, use `getattr()` Python method
    - `getattr(self, 'variable', 'default')`
  - In sum, replace `with open` :
    - `with open(getattr(self, 'file', 'data.csv'), "rU") as f:`
  - Now CSV is the default value unless overridden by setting a source file explicitly with `-a variable=value` 
  - Given a second file, `another_data.csv` :
    - `scrapy crawl fromcsv -a file=another_data.csv -o output.csv`



# Deploying to Scrapinghub

1. \+ Service
2. Scrapy Cloud -> Project Naming -> Create
3. Open Project -> menu on the left [JOBS, SPIDERS, COLLECTIONS, USAGE, REPORTS, ACTIVITY, PERIODIC JOBS, SETTINGS]
4. Setting -> Scrapy Deploy -> COPY data into project's `scrapy.cfg`
5. pip install shub 
6. shub login (with API keys)
7. shub deploy -> `Run your spiders at https://dash.scrapinghub.com/p/28814/`
8. SPIDERS -> spiders uploaded
9. Schedule -> view all info or Stop

### Programmatic Access to Scrapinghub Jobs/Data

Inspecting URL of jobs and spiders to understand entry points.

`curl -u <API>: https://storage.scrapinghub.com/items/<project id>/<spider id>/<job id>`

Leave blank if prompt pass. This allows writing applications/services using Scrapinghub as data storage backend. Mindful of time limit in cloud plan.

### Scheduling Recurring Crawls

1. PERIODIC JOBS -> Add -> set-up 



# Configuration and Management

### Settings

Source code information on DEFAULT PRIORITY `scrapy/settings/default_settings.py`

Project-level setting tuning is most practical. 

Spider-level settings via `custom_settings` attribute in spider definitions per spider.

Last-minute mod pass Shell cmd `-s CLOSESPIDER_PAGECOUNT=3`

TESTING

`scrapy settings --get CONCURRENT_REQUESTS -s CONCURRENT_REQUESTS=19`

`scrapy shell -s CONCURRENT_REQUESTS=19` 

### Essential Settings

ANALYSIS

| CODE                             | DETAIL                                                       |
| -------------------------------- | ------------------------------------------------------------ |
| (Logging) `LOG_LEVEL`            | Various levels of logs based on severity: `DEBUG` -> `INFO` -> `WARNING` -> `ERROR` -> `CRITICAL` , this controls threshold of level to display. Often `INFO` as `DEBUG` can be verbose. |
| (Logging) `LOGSTATS_INTERVAL`    | Prints number of times and pages scraped per minute. It sets logging frequency default = 60 seconds. This may be too infrequent, often 5 seconds if short run. |
| (Logging) `LOG_ENABLED`          |                                                              |
| (Logging) `LOG_FILE`             | Where logs are written, unless set, it go to STDERR except if logging gets disabled to False above. |
| (Logging) `LOG_STDOUT`           | Record all of its STDOUT (e.g. "print" msg) to log by set True. |
| (Stats) `STATS_DUMP`             | Enabled as default, it dumps values from Stats Collector to log once spider done. |
| (Stats) `DOWNLOADER_STATS`       | Control wheter stats are recorded for the downloader.        |
| (Stats) `DEPTH_STATS`            | Control whether stats are collected for site depth.          |
| (Stats) `DEPTH_STATS_VERBOSE`    | Verbose log of above.                                        |
| (Stats) `STATSMAILER_RCPTS`      | A list (e.g. set to "my@gmail.com") of e-mails to send stats to when crawl done. |
| (Telnet) `TELNETCONSOLE_ENABLED` | Python shell running process enabled as default              |
| (Telnet) `TELNETCONSOLE_PORT`    | Determines ports used to connect to console. EX1: In case wanting to look on internal  status of Scrapy while running. `DEBUG: Telnet console listening on 127.0.0.1:6023:6023` means telnet is on and listening in port 6023. Now on another terminal, use telnet command to connect to it: `telnet localhost 6023` giving a Python console inside Scrapy, for inspecting components like engine using `engine` variable or `est()` for quick overview. Very useful when using remote machine: `engine.pause() .unpause() .stop()` |

PERFORMANCE

| Code                             | Detail                                                       |
| -------------------------------- | ------------------------------------------------------------ |
| `CONCURRENT_REQUESTS`            | Maximum number of requests concurrently, mostly protects server's outbound cap. |
| `CONCURRENT_REQUESTS_PER_DOMAIN` | More restrictive, protects remote servers by limiting numer of concurrent req per unique domain or IP |
| `CONCURRENT_REQUESTS_PER_IP`     | If true, the above is ignored. NOT per second, if 16 and avg req 0.25 a second then limit is 16/0.25 = 64 req per second. |
| `CONCURRENT_ITEMS`               | Max number of items per response concurrently, per request. if 16 CONCURRENT_REQUESTS and this 100 => 1600 items concurrently wriing to DB, etc. |
| `DOWNLOAD_TIMEOUT`               | Time waited before canceling request. 180 seconds as default, seemingly excessive, advised reduction to 10 seconds. |
| `DOWNLOAD_DELAY`                 | Default to 0, mod to apply conservative download speed using this. A site might use FREQUENCY REQUEST` detect bot. |
| `RANDOMIZE_DOWNLOAD_DELAY`       | If above true, this enabled to +- 50% on delay               |
| `DNSCAHCE_ENABLED`               | For faster DNS lookups, an in-memory DNS cache is enabled by default. |

CLOSING

| Code                     | Detail                                                       |
| ------------------------ | ------------------------------------------------------------ |
| `CLOSESPIDER_ERRORCOUNT` | Auto-stop when conditions met. Often set while running spider in SHELL for testing |
| `CLOSESPIDER_ITEMCOUNT`  |                                                              |
| `CLOSESPIDER_PAGECOUNT`  |                                                              |
| `CLOSESPIDER_TIMEOUT`    | In seconds                                                   |

HTTP CACHE

| Code                          | Detail                                                       |
| ----------------------------- | ------------------------------------------------------------ |
| `HTTPCACHE_ENABLED`           | The `HttpCacheMiddleware` deactivated by default gives low-level cache for HTTP req/res |
| `HTTPCACHE_DIR`               | Relative path to project root                                |
| `HTTPCACHE_POLICY`            | If = `scrapy.contrib.httpcache.RFC2616Policy` enables way more sophy caching policy respecting sits hints according to RFC2616. (above two also True) |
| `HTTPCACHE_STORAGE`           | `scrapy.contrib.httpcache.DbmCacheStorage`                   |
| `HTTPCACHE_DBM_MODULE`        | Adjusting (defaults to anydbm)                               |
| `HTTPCACHE_EXPIRATION_SECS`   |                                                              |
| `HTTPCACHE_IGNORE_HTTP_CODES` |                                                              |
| `HTTPCACHE_IGNORE_MISSING`    |                                                              |
| `HTTPCACHE_IGNORE_SCHEMES`    |                                                              |
| `HTTPCACHE_GZIP`              |                                                              |

CRAWLING STYLE

| Code                      | Detail                                                       |
| ------------------------- | ------------------------------------------------------------ |
| `DEPTH_LIMIT`             | Max depth 0 meaning no limit.                                |
| `DEPTH_PRIORITY`          | This alows BREADTH FIRST Crawl by setting this to positive number changing from LIFO to FIFO: `DEPTH_PRIORITY = 1` (useful for news site where recent data best use FIFO method with `DEPTH_LIMIT = 3` might allow quick scan latest news on portal) |
| `SCHEDULER_DISK_QUEUE`    | Following above example: `= scrapy.squeue.PickleFileDiskQueue` |
| `SCHEDULER_MEMORY_QUEUE`  | Following above example: `= scrapy.squeue.FifoMemoryQueue`   |
| `ROBOTSTXT_OBEY`          |                                                              |
| `COOKIES_ENABLED`         | `CookiesMiddleware` takes care of all cookie-wise operations, enabling others to log in etc. If prefer more 'stealth' crawling, disable this. |
| `REFERER_ENABLED`         | Default to True enabling populating Referer headers, defined with `DEFAULT_REQUEST_HEADERS` useful for weird sites banning unless showing particular request headers !! |
| `USER_AGENT`              |                                                              |
| `DEFAULT_REQUEST_HEADERS` | Set along with Referer Headers above.                        |

FEEDS

| CODE                 | DETAIL                                                       |
| -------------------- | ------------------------------------------------------------ |
| `FEED_URI`           | `scrapy crawl fast -o "%(name)s_%(time)s.jl"` will auto-name output file. Custom variable defined in spider also allowed `%(foo)s` if foo defined. This is also set for S3, FTP. (e.g. `=s3://mybucket/file.json` along with AWS settings below) |
| `FEED_FORMAT`        | Auto-assigned based on URI extension, or set here.           |
| `FEED_STORE_EMPTY`   | Bool for empty feed.                                         |
| `FEED_EXPORT_FILEDS` | Filter esp. CSV with fixed header columns if need.           |
| `FEED_URI_PARAMS`    | Define function to postprocess any parmas to URI             |

MEDIA DOWNLOAD

| CODE                  | DETAIL                                                       |
| --------------------- | ------------------------------------------------------------ |
| `IMAGES_STORE`        | Directory stored (project root relative path) URLs for images for each ITEM should be in its `image_url` FIELD (can be overridden by `IMAGES_URLS_FIELD`) |
| `IMAGES_EXPIRES`      |                                                              |
| `IMAGES_THUMBS`       | E.g. one icon-sized and one medium size per image            |
| `IMAGES_URLS_FIELD`   |                                                              |
| `IMAGES_RESULT_FIELD` | Overrides `image` FIELD filenames                            |
| `IMAGES_MIN_HEIGHT`   |                                                              |
| `IMAGES_MIN_WIDTH`    |                                                              |
| `FILES_STORE`         | Other media, same style as Image. Both can be set at once.   |
| `FILES_EXPIRES`       |                                                              |
| `FILES_URLS_FIELD`    |                                                              |
| `FILES_RESULT_FIELD`  |                                                              |

Example - downloading images

To use image functions - `pip install image` ; to enable IMAGE PIPELINE, edit projects' `settings.py` add below.

```python
ITEM_PIPELINES = {
    ...
    'scrapy.pipelines.images.ImagesPipeline': 1,
}
IMAGES_STORE = 'images'
IMAGES_THUMBS = { 'small': (30, 30) }
```

Already have an `image_urls` field set to `Item`, so run :

`scrapy crawl fast -s CLOSESPIDER_ITEMCOUNT=90` 

AWS

| CODE                    | DETAL                                                        |
| ----------------------- | ------------------------------------------------------------ |
| `AWS_ACCESS_KEY_ID`     | Used as: When download URL start with s3:// instead of http:// etc, or s3:// path to store files with media pipelines and store output Item feeds onto s3:// directory |
| `AWS_SECRET_ACCESS_KEY` |                                                              |

PROXY

| CODE          | DETAIL                                                       |
| ------------- | ------------------------------------------------------------ |
| `http_proxy`  | `HttpProxyMiddleware` uses these settings in accordance with Unix's convention, enabled as default. |
| `https_proxy` |                                                              |
| `no_proxy`    |                                                              |

Example - Using proxies and Crawlera's clever proxy

DynDNS (or similar service) provides a free online tool to check your current IP, using Shell making a request to checkip.dyndns.org to see:

`scrapy shell http://checkip.dyndns.org`

Inside `response.body` see Current IP Address:

To start proxying requests, exit shell and use `export` command to set new proxy. Test free proxy by search through HMA's public proxy list (http://proxylist.hidemyass.com/), e.g. assuming from lsit a 10.10.1.1 and port 80

`env | grep http_proxy`

Should have nothing set, then

`export http_proxy=http://10.10.1.1:80`

Rerun Shell will see new IP.  Crawlera is Scrapy official service augmented by smart configurations

`export http_proxy=myusername:password@proxy.crawlera.com:8010`

### Further settings

- **Project**

  - housekeeping for specific project, `BOT_NAME`, `SPIDER_MODULES`, etc. Might be useful for project productivity. There are also two ENV variables, `SCRAPY_SETTINGS_MODULE` and `SCRAY_PROJECT` alowing fine tune like Django project integration. `scrapy.cfg` also allows adjusting name of settings module. 

- **Extending**

  - Allowing mod almost every aspect of Scrapy, the KING of them is `ITEM_PIPELINES` which allows use of Item Processing PIpelines. `COMMANDS_MODULE` allows adding custom commands, e.g. assuming added a `properties/hi.py` with 

    - ```python
      from scrapy.commands import ScrapyCommand
      
      class Command(ScrapyCommand):
          default_settings = {'LOG_ENABLED': False}
          def run(self, args, opts):
              print("hello")
              
      # Inside settings.py
      COMMANDS_MODULE = 'properties.hi'
      ```

  - Soon as adding `COMMANDS_MODULE='properties.hi'` in `settings.py`, it activates this command showing up in help and run with `scrapy hi`. The settings defined in `default_settings` get merged into a project's settings overriding defaults but with lower priority to that defined in `settings.py` or in Shell CLI line.

  - Scrapy uses `_BASE` dictionaries (e.g. `FEED_EXPORTERD_BASE`) to store default values for various framework extensions and then allows customising in `settings.py` and CLI by setting their non-`_BASE` version (e.g. `FEED_EXPORTERS`)

  - `DOWNLOADERS` or `SCHEULER` which hold package/class names for essential parts of system, potentially to be inherited from default downloader (`scrapy.core.downloader.Downloader`), overload a few methods, then set custom class on `DOWNLOADER` setting - allowing experimenting features and eases automated testing, if good comprehension.

- **Downloading**

  - `RETRY_*, REDIRECT_*, METAREFRESH_*` configure the Retry, Redirect and Meta-Refresh middlewares. E.g. `REDIRECT_PRIORITY_ADJUST = 2` MEANS PER REDIRECT, NEW REQUEST WILL BE SCHEDULED AFTER all non-redirected requests get served, and `REDIRECT_MAX_TIMES = 20` means after 20 redirects the downloader will give up and return whatever done. Be aware of these settings in case crawling some ill-cased sites, but default values will serve mostly. Same applies to `HTTPERROR_ALLOWED_CODES, URLLENGTH_LIMIT`

- **Autothrottle**

  - `AUTOTHROTTLE_*` configures itself. But in practice, it tends to be somewhat conservative and difficult to tune, as it suses download latencies to gauge how loaded and target server are and adjusts delay accordinly. If having hard finding best value for `DOWNLOAD_DELAY` (default = 0), should find this module useful.

- **Memory**

  - `MEMUSAGE_*` enables and configures memory which shuts down spider if exceeding limit, useful in shared ENV where processes obey resources; more often, it's useful to receive just its warning e-mail by disabling the shut down by `MENUSAGE_LIMIT_MB = 0`. (ONLY APPLICABLE IN UNIX-LIKE OS)
  - `MEMDEBUG_ENABLED, MEMDEBUG_NOTIFY` configures debugger printing number of live references on spider close, overall, chasing memory leaks is NOT fun or easy, most importantly keeping crawls relatively short, batched, and accords with server's capacity. (E.g. no good reason to run batches of more than a few thousands pages or more than a few minutes long)

- **Logging and Debugging**

  - `LOG_ENCODING, LOG_DATEFORMAT, LOG_FORMAT` fine tune logging and useful in using log-management solutions, like Splunk, Logstash. `DUPEFILTER_DEBUG, COOKIE_DEBUG` help debug relatively complex cases where exit unsual requests or lost sessions.



# Programming Scrapy

Up to here, spiders wrote have main task in defining ways crawling and extracting. Beyond Spiders, Scrapy gives mechanisms allowing fine-tune most aspects of its functionalities, such as facing:

1. Copy and paste lots of code among spiders of same project. Repeated code is more related to data (performing calculations on fields) rather than data sources
2. Having to write scripts postprocessing ITEM to drop duplicate entries or calculating values
3. Having repeated code across projects to deal with infrastructur. E.g. need to log in and transfer files to proprietary repositories, add ITEM to DB, or trigger postprocessing operations when crawls complete

Scrapy developers designed its architecture in ways allowing customisation, such as engine powering Scrapy **TWISTED**

### SCRAPY IS A TWISTED APPLICATION

Twisted Python Framework is unusual becuase it's event-driven and encourages writing asynchronous code. 

DO NOT write BLOCK code:

- Code that accesses files, databases or the Web
- Code that spawns new processes and consumes their output, like running shell CLI
- Code that performs hacky system-level operations, like waiting for system queues

Twisted gives methods allowing performing all these and more without blocking code execution.

> Imaging a typical synchronous scrapping application having 4 threads and, at any moment, 3 of them are blocked waiting for responses, 1 of them blocked performing a database write access to persist and ITEM. At any moment, it's quite unlikely to find a general-purpose thread of a scrapping app doing anything else but waiting for some blocking to pass. When blocking passes, some computations may take place for a few microseconds and then threads block again on other blocking ops likely lasting a few ms. Overall the server is not idle as it runs tens of apps utilising thousands of threads, thus after some careful tuning, CPUs remain reasonably utilised.
>
> MULTI-THREADING (4 threads)
>
> - Thread 1: blocked on web request #330
> - Thread 2: blocked on database access #79
> - Thread 3: blocked on web request #330
> - Thread 4: blocked on web request #312
>
> TWISTED (1 thread)
>
> - Thread 1: blocked waiting for any of the resources to free up
>   - Hanging : R329, D79, R330, R312, F32, ... 1000's more...
>
> Twisted approach favours using a single thread as possible, using modern OS I/O multiplexing functions (`select(), poll(), epoll()`) as HANGER, returns at once. BUT not the actual value but a hook, i.e. `deferred = i_dont_block()`, where hang whatever functionality wishign to run whenever value becomes free. Twisted application is made of chains of such deferred ops. Since single-threaded, no suffering costs of context switches and save resources (like memory) that extra threads require. Autrement dit, using this nonblocking infrastructure, gets similar performance if having thousands of threads.
>
> OS developers have been optimising thread ops for decades to make fast. The performance arguments is not as strong, but certainly writing correct thread-safe code for complex apps very hard. Mind framework change in thinking in deferred/callback, Twisted code significantly simpler than threaded code. `inlineCallbacks` generator utility makes code even simpler. 
>
> NOTE: arguably, the most successful nonblocking I/O system until now is Node.js, mainly for its high performance/concurrency. Every Node.js app uses just nonblocking APIs.

### DEFERREDS AND DEFERRED CHAINS

Deferreds are most essential mechanism Twisted offers to help write asynchronous code. APIs use deferreds to allow definig sequences of actions to be called when certain evens occur.

```python
from twisted.internet import defer

d = defer.Deferred()
d.called
# False

d.callback(3)
d.called
# True

d.result
# 3
```

See that Deferred is at core a thing representing a value that hangs, when fire d called it's called state becomes True, result attribute is set to value set on callback.

```python
d = defer.Deferred()
def foo(v):
    print("foo called")
    return v+1

d.addCallback(foo)
d.called
# Flase
d.callback(3)
# foo called
d.called
# True
d.result
# 4
```

The most powerful feature of deferred is that we can chain other ops to be called when a value is set. Add a `foo()` func as callabck of d.

### Understanding Twisted and nonblocking I/O

```
# ~*~ Twisted - A Python tale ~*~

from time import sleep

# Hello, I'm a developer and I mainly setup Wordpress.
def install_wordpress(customer):
	# Our hosting company Threads Ltd. is bad. I start installation and ...
	print("Start installation for", customer)
	# ...then wait till the installation finishes successfully. It is 
	# boring and I'm speding most of my time waiting while consuming
	# resources (RAM and CPU cycles). It's because the process is BLOCKING
	sleep(3)
	print("All done for", customer)
	
# I do this all day for our customers
def developer_day(customers):
	for customer in customers:
		install_wordpress(customer)
		
developer_day( ["Bill", "Elon", "Steve", "Mark"])

# Let's run it
$ ./deferreds.py 1
...
* Elasped time: 12.03 seconds
```

What gotten is a sequential execution. 4 customers with 3 seconds processing each means 12 overall. Doesn't scale well.

```python
from twisted.internet import reactor, defer, task

# Twisted has a slightly different approach
def schedule_install(customer):
    def schedule_install_wordpress():
        def on_done():
            print("Callback: Finished installation for", customer)
        print("Scheduling: Installation for", customer)
        return task.deferLater(reactor, 3, on_done)
	def all_done(_):
        print("All done for ", customer)
	
    # For each customer, schedule these processes on the CRM and that is all has to do
    d = schedule_install_wordpress()
    d.addCallback(all_done)
    return d
def twisted_developer_day(customers):
    work = [schedule_install(customer) for customer in customers]
    join = defer.DeferredList(work)
    join.addCallback(lambda _ : reactor.stop())
    
twisted_developer_day( ["Customer %d" % i for i in xrange(15)])

reactor.run()
```

This processes all 15 customers in parallel, 45 seconds computation in just three seconds! The trick is replacing all blocking calls to sleep() with its Twisted counterpart `task.deferLater()` and callback. 

Guide to programming scrapy:

| Problem                                                      | Solution                      |
| ------------------------------------------------------------ | ----------------------------- |
| Specific to website crawled                                  | Mod Spider                    |
| Mod or storing ITEM - domain-specific, may be reused across projects | Write an Item Pipeline        |
| Mod or dropping Request/Reponse - domain-specific, mmay be reused across projects | Write a spider middleware     |
| Executing Requests/Responses - generic, like to support some custom login scheme or a special way to handle cookies | Write a downloader middleware |
| All other problems                                           | Write an extension            |



### Example 1 - a very simple pipeline

Problem: Lots of spiders, but database need string format for indexing, changing individual spiders too much code.

Write a postprocess item pipeline:

```python
from datetime import datetime

class TidyUp(object):
    def process_item(self, item, spider):
        item['date'] = map(datetime.isoformat, item['date'])
        return item
```

Simple class with `process_item()` method. Add it in `tidyup.py` insdie `pipelines` directory.

NOTE: The placement of code is free, but a separate directory is a good idea.

Now edit `settings.py` and set

```ITEM_PIPELINES = { 'properties.pipelines.tidyup.TidyUp' : 100 }`

The number 100 on dict defines the order in which pipelines are connected. If another pipeline has a smaller number, it will process ITEM prior to this pipeline.

The resulting `date` data will be `['2015-11-08T14:47:04.148232']` as ISO string.



### Signals

Mechanism to add callbacks to events happening in system, such as when a spider opens, or when an item gets scraped. Hook to them using `crawler.signals.connect()` (see below example). There're just 11 of them and maybe the easiest way to understand in action. Below is a project having an extension hooking to all signals. Plus a Item Pipeline, one Downloader and one spider middleware, logging every method invocation. 

```python
def parse(self, response):
    for i in range(2):
        item = HooksasyncItem()
        item['name'] = "Hello %d" % i
        yield item
	raise Exception("dead")
```

On the second ITEM, configured the Item Pipeline to raise a `DropItem` exception.

This illustrates when certain signals get sent via logs:

```
$ scrapy crawl test
...many lines....

# First we get those two signals...
INFO: Extension, signals.spider_opened fired
INFO: Extension, signals.engine_started fired

# Then for each URL get a request_scheduled signal
INFO: Extension, singals.request_scheduled fired

# when downlad compltes we get
INFO: Extension, signals.response_downloaded fired
INFO: DownloaderMiddlewareprocess_response called for example.com

# Work between 
INFO: Extension, singals.response_received fired
INFO: SpiderMiddlewareprocess_spider_input called for..

# here our parse() gets called then SpiderMiddleware use
INFO: SpiderMiddlewareprocess_spider_output called for url

# For every Item going through pipelines successfullly...
INFO: Extension, signals.item_scraped fired

# For every item gets dropped using DroptItem exception
INFO: Extension, signals.item_dropped fired

# If your spider throws sth lese..
INFO: Extension, signals.spider_error fired

# ... the above process repeats for each URL
# ... till we run out of them. then..
INFO: Extension, signals.spider_idle fired

# by hooking spider_idle you can shcedule further Requests. if you dont the spider ends
INFO: Closing spider (finished)
INFO: Extension, signals.spider_closed fire

# ....stats get printed and finally engines get stopped
INFO: Extension, singal.sengine_stopped fired

```

Only 11 signals, but every scrapy default middleware is implemented using just them, so they must be sufficient. Note every signal except spider_idle, error, request, you can also return deferreds instead of actual values.

### Example 2 - an extension measuring thorugput and latencies

Built-in extension for this the Log Stats extenson (`scrapy/extensions/logstats.py`) in source code as starting point. To measure latencies, hook the `request_scheduled, response_received, item_scraped` signals. 

```python
class Latencies(object):
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    def __init__(self, crawler):
        self.crawler = crawler
        self.interval = crawler.settings.getfloat('LATENCIeS_INTERVAL')
        	if not self.interval:
                raise NotConfigured
		cs = crawler.signals
        cs.connect(self._spider_opened, signal=signals.spider_opened)
        cs.connect(self._spider_closed, signal=signals.spider_closed)
        cs.connect(self._request_scheduled, signal=signals.request_scheduled)
        cs.connect(self._request_received, signal=signals.request_received)
		cs.connect(self._item_scraped, signal=signals.item_scraped)
		self.latency, self.proc_latency, self.items = 0, 0, 0
	def _spider_opened(self, spider):
        self.task = task.LoopingCall(self._log, spider)
        self.task.start(self.interval)
        
	def _spider_closed(self, spider, reason):
        if self.task.running:
            self.task.stop()
	def _request_scheduled(self, request, spider):
        request.meta['schedule_time'] = time()
	def _reponse_received(self, response, request, spider):
        request.meta['received_time'] = time()
	def _item_scraped(self, item, response, spider):
        self.latency += time() - response.meta['schedule_time']
        self.proc_latency += time() - response.meta['received_tieme']
        self.items += 1
	def _log(self, spider):
        irate = float(self.items) / self.interval
        latency = self.latency / slef.items if self.items else 0
        proc_latency = self.proc_latency / self.items if self.items else 0
        spider.logger.info(("Scraped %d items at %.1f items/s, avg latencys: "
                           "%.2f s and avg time in pipelines: %.2f s") %
                           (self.itesm, irate, latency, proc_latency))
        self.latency, self.proc_latency, self.items = 0, 0, 0
```

The first wo methods key as they are typical. INIT middleware using Crawler object. They're every nontrivial middleware. `from_crawler(cis, craler)` is way of grabbing the crawler object. Then notice in __init__() accesing crawler.settings and raise exception if not set. Many FooBar extensions checking the corresponding FOOBAR_ENABLED setting and raise if not set or Flase. This is very common pattern allowing middleware to be included for ease in matching settings.py settings (ITEM__PIPELINES, for example) but being disabled by default, unless enableld by flag settings. Many default middleware (AutoThrottle or HttpCache) use this pattern. In this case, the extension remains disabled unless LATENCIES_INTERVAL is est

Then `__init__()` register callbacks for all signals interested in using crawler.signals.connect(), and INIT afew member vars, the rest of class deploys singals handlers. 

NOTE: by analogy to multithreaded context code this absence of mutexes in code will see single-threaded is eaiser and scales well in more complex scenarios.

Add this extension in `latencies.py` module at the same level as `settings.py`. Enable it by adding in `settings.py`

`EXTENSIONS = { 'properties.latencies.Latencies' : 500, }`

`LATENCIES_INTERVAL = 5 `

Now the running log will print INFO as desinged. 



### EXTENDING BEYOND MIDDLEWARES

Inspecting source code in `default_settings.py` will see a few class names among it. Scrapy extensively sues a dependency-injection-like mechanism allowing customisation and extension of its internal obejcts. E.g. one may want to supplort more protocls for URLs beyond files, HTTP, HTTPS, S3, and FTP that are defined in `downlaod_handlers_base` SETTING. MOST DIFFICULT PART IS TO DISCOVER WHAT THE INETERFACE FOR YOUR CUSTOM CLASSES MUST BE (I.E. WHICH MEHTODS TO IMPLEMENT) AS MOST INTERFACES ARE NOT EXPLICIT. One has to read source code and see how these classes get used. You best bet is starting with an existing implementation and altering it to your need. That said, these interfaces become more and more stable with recent versions. 



# PIPELINE Recipes

Previous on middlewares, now on pipelines by showcasing consuming REST APIs, interfaciing with DB, performing CPU-intensive tasks, and interfacing with legacy services.

### Using REST APIs

REST is a set of techs used to create modern web services. Its main pro is simpler more lightweight htan SOAP or else. Software designers see a similarity between CRUD (Create, Read, Update, Delete) that web services often provide and basic HTTP ops (GET POST PUT DELETE). Also seeing much of info required for typical web-serivce call could be compacted on a resoruce URL. e.g. http://api.mysite.com/customer/john is a resource URL alowing to identify target server (api.mysite.com), the fact that to performing ops related to customers table in taht server, and more specifically somehting taht has to do with somethe named johb (row-primary key). This plus other web oncepts like secure AUTH, being stateless, caching, XML, JSON as payload, etc provides a powerful yet simple, familiar and effortlessly cross-platform way to provide and consume web services. Quite common some of the fucntioanlity needed to use in Scrapy pipelien to be provided in the form of REST API.

USING treq

`treq` is a Pyton pkg trying to equate Python `requests` pkg for Twisted-based apps. One would prefer `treq` over Scrapy's `Request/crawler.engine.download()` API for it's equally simple, but it has perforance pros.

PIPELINE WRITING TO **ELASTICSEARCH**

Start by writing ITEM on an ES server. Perhaps begin with ES (even before MySQL) as persistence mechanism a bit unusual, but it's actually the easiest thing one ca ndo. ES can be schema-less, meaning using it without any configuration. `treq` is also enough for this case.

`curl http://es:9200` returning JSON =like data. To DELETE: `curl -XDELETE http://es:9200/proeprties`

```python
@defer.inlineCallbacks
def process_item(self, item, spider):
    data = json.dumps(dict(item), ensure_ascii=False).encode("utf8")
    yield treq.post(self.es_url, data)
```

The first two lines define a standard `process_item()` able to `yield Deferred` as illustrated before.

Third line prepares data for insertion. First convert ITEM to dicts, with encoding etc. Last line uses `post()` of `treq` to perform POST request inserting doc in ES. 

To enable pipeline, need to add it on `ITEM_PIPELIENS` setting insdie settings.py and INIT `ES_PIPELINE_URL` settings:

```python
ITEM_PIPELINES = {
    'properties.pipelines.tidyip.TidyUp': 100,
    'properties.pipelines.es.EsWriter': 800,
}
ES_PIPELINE_URL = 'http://es:9200/proeprties/property'
```

NOTE: is it a good idea to use pipelines to insert ITEM in DB? NO, often DB provide orders of magnitude more efficient ways to bulk insert entries, and we shuld definietley use them instead. This would mean bulking ITEMS and batching inserting them or performing inserts as post-processing step at ned of crawl. 

PIPELINE GEOCODES USING GOOGLE GEOCODING API

Say area names for our properties, like to geocode them, finding respoective coordinates. Google Geocoding API saves the effort of complex DB, sophisticated text mathcing and spatial computations. 

`curl "https://maps.googleapis.com/maps/api/geocode/json?sensor=false&address=london"` will return JSON of info.

Google API is accessible using same techniques as `treq` saving as `geo.py` inside `pipelines` directory:

```python
@defer.inlineCallbacks
def geocode(self, addresss):
    endpoint = 'http://web:9312/maps/api/geocode/json'
    
    params = [('address', address), ('sensor', 'false')]
    response = yield treq.get(endpoint, params=params)
    content = yield response.json()
    
    geo = content['results'][0]["geometry"]["location"]
    defer.returnValue( {"lat": geo["lat"], "lon": geo["lng"]})
```

The endpoint is for faking for faster execution, less intrusive, available offline, more predictable. You can use endpoint = actual google api URL to hi Google's servers, but keep in mind STRICT LIMIT ON REQUESTS. 

Now `process_item()` becomes a single line `item['location'] = yield self.geocode(item["address"][0])`

Enables: `ITEM_PIPELINES = { ... properties.pipelines.geo.geoPipeline': 400, ...}`



# Official Scrapy Tutorial

```python
import scrapy

class QuotesSpider(scrapy.Spider):
	name = "quotes"
	allowed_domains = ["toscrape.com"]
	start_urls = ['http://quotes.toscrape.com']

	# Version1: Scraping by pages
	def parse(self, response):
		self.log(f'I just visited: {response.url}')
		for quote in response.css('div.quote'):
			item = {
				'author_name': quote.css('small.author::text').extract_first(),
				'text': quote.css('span.text::text').extract_first(),
				'tags': quote.css('a.tag::text').extract(),
			}
			yield item
		# follow pagination link
		next_page_url = response.css('li.next > a::attr(href)').extract_first()
		# stop at null page link
		if next_page_url:
			# response.urljoin('relative path') joins with response.url('abs path')
			next_page_url = response.urljoin(next_page_url)
			yield scrapy.Request(url=next_page_url, callback=self.parse)

	# Version2: Scraping through links
	def parse(self, response):
		# collecting all links wishing to click on a page
		# DEFAULT DUPEFILTER set to ignore duplicate pages
		urls = response.css('div.quote > span > a::attr(href)').extract()
		for url in urls:
			url = response.urljoin(url)
			yield scrapy.Request(url=url, callback=self.parse_details) # callback to be defined

		# follow pagination links
		# same as above

	def parse_details(self, response):
		yield {
			'name': response.css('h3.author-title::text').extract_first()
			'birth': response.css('span.author-born-date::text').extract_first()
		}

# Version3: Scraping Infinite Scrolling Pages; finding APIs powering AJAX-based inf-scroll
"""Concept
Using DevTool to inspect network as scrolling happens returning AJAX powered, mostly, JSON files;
explorable inside DevTool;
Preview by json library tools:
response.text 	:revealing JSON format
print(response.text) 	:readable format
import json
data = json.loads(response.text)
data.keys()		:prints keys
data['quotes'][0] 	:first element of quotes, a dict
data['quotes'][0]['author']['name'] 	:lowest-level data
"""
import json

class QuoteScrollSpider(scrapy.Spider):
	name = "quotes-scroll"
	api_url = 'http://quotes.toscrape.com/api/quotes?page={}'
	start_urls = [api_url.format(1)] # KEY step

	def parse(self, response):
		data = json.loads(response.text)
		for quote in data['quotes']:
			yield {
			'author': quote['author']['name'],
			'text': quote['text']
			'tags': quote['tags']
			}
		if data['has_next']:
			next_page = data['page'] + 1
			yield scrapy.Request(url=self.api_url.format(next_page), callback=self.parse)

# Version4: Submitting Forms - POST requests such as logins
"""Concept
Network-inspect requests at login reveals POST request such as login with value like username, password;
In example case, there's a hidden input 'type="hidden" name="carf_token"', inspect via page-source-code;
Its value is often HASHed of something, which in case is randomised per page load;
Solution: submit form with user/pass + carf_token scraped at page-load
"""
class LoginSpider(scrapy.Spider):
	name = 'loging-spider'
	login_url = 'http://quotes.toscrape.com/login'
	start_urls = [login_url]

	def parse(self, response):
		# extract the CSRF token (selector depending on context)
		token = response.css('input[name='csrf_token']::attr(value)').extract_first()
		# create a python dict with form values
		data = {
			'csrf_token': token,
			'username': 'whatever',
			'password': 'whatever',
		}
		# submit a POST request to web (url may differ from login page)
		yield scrapy.FormRequest(url=self.login_url, formdata=data, callback=self.parse.quotes)


	def parse_quotes(self, response):
		"""Parse the main page after the spider logged in"""
		for q in response.css('div.quote'):
			yield {
				'author_name': q.css('small.author::text').extract_first(),
				'author_url': q.css('small.author ~ a[href*="goodreads.com"]::attr(href)').extract_first()
			}

# Even simpler way is to use FormRequest.from_request() directly parsing hidden field!!
"""
This method reads the response object and creates a FormRequest that automatically includes all the pre-filled values from the form, along with the hidden ones. 
This is how our spider's parse_tags() method looks:
So, whenever you are dealing with forms containing some hidden fields and pre-filled values, use the from_response method because your code will look much cleaner.
"""
	def parse_tags(self, response):
	    for tag in response.css('select#tag > option ::attr(value)').extract():
	        yield scrapy.FormRequest.from_response(
	            response,
	            formdata={'tag': tag},
	            callback=self.parse_results,
	        )


# Version5: Scraping JS pages with Splash: scraping JS-based webs using Scrapy + Splash
"""Concept
JS-based pages returns only static HTML when scraped by Scrapy, or whatever present inspected via PAGE SOURCE;
Active inspection will show JS called content, which is not scraped by Scrapy;

Splash - JS Engine
	docker pull scrapinghub/splash
	docker run -p 8050:8050 scrapinghub/splash
		# now splash is LISTENING to the local 8050 port
		# so Spider can REQUEST to it, Splash fetches the page, execute JS code on it, then returning rendered pages to spider
	pip install scrapy-splash

Need to config SETTING:

	DOWNLOADER_MIDDLEWARES = {
	    'scrapy_splash.SplashCookiesMiddleware': 723,
	    'scrapy_splash.SplashMiddleware': 725,
	    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
	}

The middleware needs to take precedence over HttpProxyMiddleware,
which by default is at position 750, so we set the middleware positions to numbers below 750.

You then need to set the SPLASH_URL setting in your project's settings.py:
	SPLASH_URL = 'http://localhost:8050/'

Don’t forget, if you’re using a Docker Machine on OS X or Windows, you will need to set this to the IP address of Docker’s virtual machine, e.g.:
	SPLASH_URL = 'http://192.168.59.103:8050/'

Enable SplashDeduplicateArgsMiddleware to support cache_args feature: it allows to save disk space by not storing duplicate Splash arguments multiple times in a disk request queue. If Splash 2.1+ is used the middleware also allows to save network traffic by not sending these duplicate arguments to Splash server multiple times.
	SPIDER_MIDDLEWARES = {
    'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
	}

Scrapy currently doesn’t provide a way to override request fingerprints calculation globally, so you will also have to set a custom DUPEFILTER_CLASS and a custom cache storage backend:
	DUPEFILTER_CLASS = 'scrapy_splash.SplashAwareDupeFilter'
	HTTPCACHE_STORAGE = 'scrapy_splash.SplashAwareFSCacheStorage'

If you already use another cache storage backend, you will need to subclass it and replace all calls to scrapy.util.request.request_fingerprint with scrapy_splash.splash_request_fingerprint.

Now that the Splash middleware is enabled, you can use SplashRequest in place of scrapy.Request to render pages with Splash.

For full list of arguments in HTTP API doc: http://splash.readthedocs.org/en/latest/api.html
By default the endpoint is set to 'render.json' but here overridden and set to 'render.html' for HTML response
"""

import scrapy
from scrapy_splash import SplashRequest

class QuotesJSSpider(scrapy.Spider):
	name = 'quotesjs'

	def start_request(self):
		yield SplashRequest(
			url='http://quotes.toscrape.com/js',
			callback=self.parse,
			)
	def parse(self, response):
		for quote in response.css(...)
		yield ...

        

"""
**Running Custom JS**
Sometimes you need to press a buttom or close a modal to view pape properly.
Splash lets run custom JS code within the context of web page:

1) Using js_source Parameter
Code is run after page loaded but before page rendered, allowing use of JS code
to modify page being rendered:

EX: render page and mod its title dynamically
	yield SplashRequest(
	    'http://example.com',
	    endpoint='render.html',
	    args={'js_source': 'document.title="My Title";'},
	)

2) Splash Scripts
Splash supports LUA scripts via execute endpoint. Preferred way for preload libraries
choosing when to execute JS and retrieve output.

Sample script:
	function main(splash)
	    assert(splash:go(splash.args.url))
	    splash:wait(0.5)
	    local title = splash:evaljs("document.title")
	    return {title=title}
	end

Need to send script to execute endpoint, in lua_source arguments, returning a JSON object having title:
	{
	    "title": "Some title"
	}

Every script needs a main func to act as entry point. Able to return lua table be rendred as JSON, as here.
Using splash:go function to tell Splash to visit the URL, splash:evaljs function lets run JS within page context,
but if no need result then use splash:runjs instead

Test Splash scripts in browser at instance's index page set above, 
For mouse-click function: using splash:mouse_click

	function main(splash)
	    assert(splash:go(splash.args.url))
	    local get_dimensions = splash:jsfunc([[
	        function () {
	            var rect = document.getElementById('button').getClientRects()[0];
	            return {"x": rect.left, "y": rect.top}
	        }
	    ]])
	    splash:set_viewport_full()
	    splash:wait(0.1)
	    local dimensions = get_dimensions()
	    splash:mouse_click(dimensions.x, dimensions.y)
	    -- Wait split second to allow event to propagate.
	    splash:wait(0.1)
	    return splash:html()
	end

Here splash:jsfunc defined to return element coordinates, visible by splash:set_viewport_full, click element and return HTML
"""


"""
Run Spider on Cloud: deploy, run and manage crawlers in cloud
Above are single spider.py, now build a project
	scrapy startproject quotes_crawler
Example, move one of above spiders to project and run
	scrapy crawl spiderName

Scraping Hub as Cloud
	pip install shub

	shub login # then API key
	shub deploy # requring Project ID, which is digits on URL of project page
		# review under Code & Deploys
Run -> Spider -> Job Units etc
Inspect result under Job with downloadable formats
Schedule features
Also CLI
	shub schedule quotes # run the spider
"""
```



## Udemy Video Summary
### (1) Scrapy Architecture
Root/scrapy.pyc [settings] path and [deploy] url and project folder

Under ProjectFolder: 
 - __init__.py 		:directories
 - items.py 		:item classes which be imported inside spider (via scrapy.loader.ItemLoader)
 - pipelines.py 	:process_item and related processing as pipelines (init in settings.py -> pipeline class inserted)

### (2) Avoiding Ban
	a) DOWNLOAD_DELAY or via time.sleep(random.randrange(1,3)) at end of code
	b) USER_AGENT 
	c) Proxies scrapy-proxies package or use VPN
	d) Professional work using ScrapingHub
	e) Be mindful of regulation and rights

### (3) Runspider for standalone scripting
	- Without use of ITEM/PIPELINE etc
	- Print out or yield result

### (4) scrapy.spiders.CrawlSpider has more functions such as RULE, which need importing itself
	rules = (Rule(LinkExtractor(allow=('music'), deny_domains=('google.com')), callback='parse_be_defined', follow=False),) 
### (5) scrapy.http.Request method used under ordinary parse(self,response) WITHOUT callback to loop through new Request(url) to parse() !!
### (6) relative URL fixing, e.g. images
	- Inspect HTML //img/@src to see relative path, e.g. ../../path/to/image.jpg
	- Replace ../../ with actual image URL
		image_url = image_url.replace('../..', 'http://missingPath')

### (7) Define functions to extract well-formated datapoints, e.g. tables of data
	def product_info(response, value)
		return response.xpath('//th[text()="' + value + '"]/following-sibling::td/text()').extract_first()
	then use it to extract and save into ITEM key-value pairs

### EXAMPLE CODE

```python
from scrapy import Spider
from scrapy.http import Request

def product_info(response, value):
    ...

class BooksSpider(Spider):
    name
    allowed_domains
    start_urls

	def parse(self, resonse):
		books = response.xpath('//h3/a/@href').extract()
		for book in books:
			abs_url = response.urljoin(book)
			yield Request(abs_url, callback=self.parse_book)
		# process next page
		next_url = response.xpath('//a[text()="next"]/@href').extract_first()
		abs_next_url = response.urljoin(next_url)
		yield Request(abs_next_url)
	
	def parse_book(self, response):
		title = response.css('h1::text').extract_first()
		price = response.xpath('//*[@class="price_color"]/text()').extract_first()
		image_url (as above)
		rating = response.xpath('//*[contains(@class, "star-rating")]/@class').extract_first()
		rating = rating.replace('star-rating ', '')
	
		description = response.xpath('//*[@id="product_description"]/following-sibling::p/text()').extract_first()
	
		# product table as above
		upc = product_info(response, 'UPC')
```
-------------------------------------------------
### (8) Arguments: e.g. isolating 'book categories'
Update above code
​	REMOVE start_url WITH:
```python
def __init__(self, category): # constructor!
self.start_urls = [category] 

THIS CREATES A ARGUMENT-ABLE FOR __INIT__
used in Shell: 
scrapy crawl bookspider -a category="category_specific_URL"
```

### (9) Scrapy Functions: executed at end of crawling
Anything needed to run, cleaning, sending, etc. Defined inside Spider.py

```python
# EX overriding output filename.csv a function postprocessing 

import os
import glob

	def close(self, reason):
		csv_file = max(glob.iglob('*.csv'), keys=os.path.getctime)
		os.rename(csv_file, 'foobar.csv')

>> ... -o item.csv
```

### (10) Feeding

... -o items.csv/json/xml # items can be whatever

### (11) Image Download via built-in ImagesPipeline
Best first define Item class in items.py with all required datapoints + image
Then change settings.py
```python
ITEM_PIPELINES = {
	'scrapy.pipelines.images.ImagesPipeline': 1,
}
IMAGES_STORE = 'local/folder'
```
Then add ItemLoader and items.definedClass in spider.py