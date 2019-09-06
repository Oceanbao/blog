---
title: "hugo shortcode demo"
date: 2018-03-16T20:18:53-05:00
showDate: true
draft: false
tags: ["code"]
---

Shortcodes are pre-made templates converting HTML codes into shortcodes in hugo. Here illustrates some:

Permalink in hugo:
[Qui]({{< relref "/qui.md">}})


Code higlighting:
{{< highlight html >}}

<section id="main">
  <div>
   <h1 id="title">{{ .Title }}</h1>
    {{ range .Pages }}
        {{ .Render "summary"}}
    {{ end }}
  </div>
</section>
{{< /highlight >}}

firgure or image:
{{< figure src="https://unsplash.com/photos/1GUwkryQiro" title="title here" width=50, height=50, caption="caption here">}}



