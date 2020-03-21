---
layout: default
title: Kevin Nowland Code Blog
---

# Code

Hello, World

<ul>
  {% for post in site.categories.code %}
    <li>
      <a href="{{ post.url }}">{{ post.date | date_to_string }} - {{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
