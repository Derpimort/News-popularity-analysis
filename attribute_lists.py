#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:11:58 2020

@author: darp_lord
"""
import re

TITLE_L=[("title",), 
                 ("meta", {"property": re.compile(".*title.*")}),
                 ("meta", {"name": re.compile(".*title.*")})]
KEYWORD_L=[("meta", {"property": re.compile(".*keyword.*")}),
           ("meta", {"name": re.compile(".*keyword.*")})]
DESC_L=[("meta", {"property": re.compile(".*desc.*")}),
        ("meta", {"name": re.compile(".*desc.*")})]
AUTHOR_L=[("meta", {"property": re.compile(".*author.*")}),
          ("meta", {"name": re.compile(".*author.*")}),
          ("meta", {"property": re.compile(".*publisher.*")}),
          ("meta", {"name": re.compile(".*publisher.*")})]
PUBLISHED_L=[("meta", {"property": re.compile(".*published.*")}),
             ("meta", {"name": re.compile(".*published.*")}),
             ("meta", {"property": re.compile(".*created.*")}),
             ("meta", {"name": re.compile(".*created.*")}),
             ("meta", {"property": re.compile(".*date.*")}),
             ("meta", {"name": re.compile(".*date.*")}),
             ("meta", {"property": re.compile(".*time.*")}),
             ("meta", {"name": re.compile(".*time.*")})]
CONTENT_L=[("div", {"class": re.compile(".*article.*")}),
           ("section", {"class": re.compile(".*article.*")}),
           ("section",),
           ("div", {"class": re.compile(".*content.*")})]