---
layout: post
mathjax: true
comments: true
title: Chinese Characters Decoding using BeastifulSoup
Category: Data Science
---

When scraping Chinese website using python (Windows 10 system) and requests, it usually returns "gb2312" code for Chinese characters. However, if one does not declare the encoding of requests, it uses its default encoding, which is __not__ "gb2312". The following example provides a solution to scrape Chinese contents and save them as Unicode encoding in a data file.


```
import requests

from bs4 import BeautifulSoup

# set sys default enconding to be unicode utf-8

import sys

sys.setdefaultencoding("utf-8")


# website scraping with request

url_to_scrape = 'http://www.mitbbs.com'



readOut = requests.get(url_to_scrape)

# in request, there is a method to search/get the real encoding of the website which is

# apparent_endcoding, so one need to set the encoding to be the apparent_encoding

readOut.encoding = readOut.apparent_encoding


# use beautifulsoup to get the text information

textSoup = BeautifulSoup(readOut.text, "lxml")


# now when printing out the content in the textSoup, you will get the right display of Chinese characters.

print(textSoup.title.string)


# with the sys default encoding to be uft-8, it will give the right display of Chinese characters in the txt file.

fileToWrite = open("fileToWrite.txt", "w")

fileToWrite.write("%s" %textSoup.title.string)

fileToWrite.close()
```
